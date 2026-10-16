"""
Screen-record the real RViz2 while ``ros2 bag play`` replays the bag.

This is the pixel-exact alternative to ``render``: it shows the URDF meshes
and every RViz plugin, but runs in real time (or ``--rate`` times it).
Everything runs on an isolated ROS_DOMAIN_ID so the replayed topics never
reach a live robot on domain 19.

RViz runs on a private X server -- Xvfb if installed (invisible), else
Xephyr (a nested server in its own desktop window) -- and ffmpeg records
that server's framebuffer. Recording the real desktop instead (``--display
desktop``) is opt-in only: x11grab copies screen pixels, so whatever covers
the RViz window (a video call, a browser) ends up in the video.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from . import bag_io

DESCRIPTION_PUBLISHER = r'''
import sys, rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
rclpy.init()
node = rclpy.create_node("bag_converter_robot_description")
pub = node.create_publisher(String, "/robot_description",
                            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
text = open(sys.argv[1], encoding="utf-8").read()
pub.publish(String(data=text))
timer = node.create_timer(1.0, lambda: pub.publish(String(data=text)))
rclpy.spin(node)
'''


def default_urdf():
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("mecanumbot_description"),
                         "urdf", "mecanumbot.urdf")
        if os.path.isfile(p):
            return p
    except Exception:  # noqa: BLE001
        pass
    p = os.path.expanduser(
        "~/Documents/mecanumbot_ws/src/mecanumbot/mecanumbot_description/urdf/mecanumbot.urdf")
    return p if os.path.isfile(p) else None


def prefixed_urdf(path, prefix):
    """Return a temp copy of the URDF with every link name prefixed (e.g. ``mecanumbot/``)."""
    if not prefix:
        return path
    text = open(path, encoding="utf-8").read()
    text = re.sub(r'(<link\s+name=")([^"]+)"', lambda m: f'{m.group(1)}{prefix}{m.group(2)}"',
                  text)
    text = re.sub(r'(<(?:parent|child)\s+link=")([^"]+)"',
                  lambda m: f'{m.group(1)}{prefix}{m.group(2)}"', text)
    fd, out = tempfile.mkstemp(suffix=".urdf", prefix="bag_converter_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out


def _window_geometry(display, title_pattern="RViz", timeout=30.0):
    """Return (x, y, w, h) of the first window whose name matches, polling until timeout."""
    deadline = time.time() + timeout
    env = dict(os.environ, DISPLAY=display)
    while time.time() < deadline:
        tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True,
                              env=env).stdout
        for line in tree.splitlines():
            m = re.match(r'\s*(0x[0-9a-f]+) "([^"]*)"', line)
            if m and title_pattern in m.group(2):
                info = subprocess.run(["xwininfo", "-id", m.group(1)], capture_output=True,
                                      text=True, env=env).stdout
                vals = {}
                for key in ("Absolute upper-left X", "Absolute upper-left Y", "Width", "Height"):
                    mm = re.search(rf"{key}:\s+(-?\d+)", info)
                    vals[key] = int(mm.group(1)) if mm else 0
                if vals["Width"] > 200 and vals["Height"] > 200:
                    return (max(0, vals["Absolute upper-left X"]),
                            max(0, vals["Absolute upper-left Y"]),
                            vals["Width"] - vals["Width"] % 2,
                            vals["Height"] - vals["Height"] % 2)
        time.sleep(0.5)
    return None


def _stop(proc, sig=signal.SIGINT, timeout=10):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def latched_topics(bag):
    """Return topics recorded from transient-local publishers (durability 1 in metadata)."""
    out = []
    for entry in bag_io.read_metadata(bag).get("topics_with_message_count", []) or []:
        meta = entry.get("topic_metadata", {})
        if re.search(r"durability:\s*1\b", meta.get("offered_qos_profiles", "") or ""):
            out.append(meta["name"])
    return out


def _pick_server(choice):
    if choice == "desktop":
        return "desktop"
    if choice in ("auto", "xvfb") and shutil.which("Xvfb"):
        return "xvfb"
    if choice in ("auto", "xephyr") and shutil.which("Xephyr"):
        return "xephyr"
    raise SystemExit(
        f"no private X server for --display {choice}: install one "
        "(sudo apt install xvfb), or pass --display desktop to record the real screen")


def _free_display():
    for n in range(90, 200):
        if not os.path.exists(f"/tmp/.X11-unix/X{n}") and not os.path.exists(f"/tmp/.X{n}-lock"):
            return f":{n}"
    raise SystemExit("no free X display number in :90-:199")


def _wait_for_display(disp, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok = subprocess.run(["xwininfo", "-root"], env=dict(os.environ, DISPLAY=disp),
                            capture_output=True).returncode == 0
        if ok:
            return True
        time.sleep(0.2)
    return False


def capture_bag(bag, output, config_path, rate=1.0, fps=30, domain_id=77, startup_wait=8.0,
                urdf=None, frame_prefix="auto", start_s=None, duration_s=None,
                virtual_display="auto", size=(1850, 1136), crf=20):
    bag = bag_io.resolve_bag(bag)
    for tool in ("ffmpeg", "rviz2", "ros2", "xwininfo"):
        if not shutil.which(tool):
            raise SystemExit(f"capture needs `{tool}` on PATH")

    env = dict(os.environ, ROS_DOMAIN_ID=str(domain_id))
    procs = []
    tmp_files = []
    screen_w, screen_h = size[0] + 80, size[1] + 64   # room for RViz's saved window offset
    try:
        server = _pick_server(virtual_display)
        if server == "desktop":
            disp = os.environ.get("DISPLAY")
            if not disp:
                raise SystemExit("no $DISPLAY to record")
            print("      WARNING: recording the real desktop; anything covering the RViz "
                  "window will be in the video", file=sys.stderr)
        else:
            disp = _free_display()
            if server == "xvfb":
                cmd = ["Xvfb", disp, "-screen", "0", f"{screen_w}x{screen_h}x24"]
            else:
                cmd = ["Xephyr", disp, "-screen", f"{screen_w}x{screen_h}", "-title",
                       "bag_convert capture (private display)", "-no-host-grab"]
            procs.append(subprocess.Popen(cmd, start_new_session=True,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            if not _wait_for_display(disp):
                raise SystemExit(f"{server} did not start on {disp}")
            print(f"      RViz on private {server} display {disp}", file=sys.stderr)
        env["DISPLAY"] = disp

        urdf = urdf or default_urdf()
        if urdf:
            if frame_prefix == "auto":
                types = bag_io.topic_types(bag)
                frame_prefix = ""
                if "/tf_static" in types:
                    for _, msg, _ in bag_io.iter_messages(bag, ["/tf_static"], types):
                        child = msg.transforms[0].child_frame_id if msg.transforms else ""
                        if "/" in child:
                            frame_prefix = child.rsplit("/", 1)[0] + "/"
                        break
            urdf_path = prefixed_urdf(urdf, frame_prefix)
            if urdf_path != urdf:
                tmp_files.append(urdf_path)
            procs.append(subprocess.Popen(
                [sys.executable, "-c", DESCRIPTION_PUBLISHER, urdf_path], env=env,
                start_new_session=True))

        rviz = subprocess.Popen(
            ["rviz2", "-d", config_path, "--ros-args", "-p", "use_sim_time:=true"],
            env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        procs.append(rviz)
        geom = _window_geometry(disp, timeout=max(30.0, startup_wait * 3))
        if geom is None:
            err = rviz.stderr.read().decode(errors="replace")[-2000:] if rviz.poll() else ""
            raise SystemExit(f"RViz window did not appear on {disp}. {err}")
        time.sleep(startup_wait)   # let RViz finish loading plugins/meshes
        x, y, w, h = geom
        if server != "desktop":   # clip to the private screen
            w = min(w, screen_w - x) // 2 * 2
            h = min(h, screen_h - y) // 2 * 2
        print(f"      recording RViz window {w}x{h}+{x}+{y} on {disp}", file=sys.stderr)

        vf = f"setpts={rate:g}*PTS" if rate != 1.0 else "null"
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0",
             "-framerate", str(fps),
             "-video_size", f"{w}x{h}", "-i", f"{disp}+{x},{y}", "-vf", vf, "-r", str(fps),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", output],
            stdin=subprocess.PIPE, env=env, start_new_session=True)
        procs.append(ffmpeg)

        latched = latched_topics(bag)
        if start_s and latched:
            # --start-offset would skip the latched messages recorded at the bag's start
            # (/map is published once): replay those first, without /clock, so RViz has them.
            pre = ["ros2", "bag", "play", bag, "-r", "1000", "--topics", *latched]
            subprocess.run(pre, env=env, stdout=subprocess.DEVNULL, timeout=120)

        play_cmd = ["ros2", "bag", "play", bag, "--clock", "100", "-r", f"{rate:g}"]
        if start_s:
            play_cmd += ["--start-offset", f"{start_s:g}"]
        print("      " + " ".join(play_cmd), file=sys.stderr)
        player = subprocess.Popen(play_cmd, env=env, start_new_session=True,
                                  stdout=subprocess.DEVNULL)
        procs.append(player)
        try:
            if duration_s:
                player.wait(timeout=duration_s / rate)
            else:
                player.wait()
        except subprocess.TimeoutExpired:
            _stop(player)
        time.sleep(0.5)
        ffmpeg.stdin.write(b"q")
        ffmpeg.stdin.flush()
        ffmpeg.wait(timeout=30)
        print(f"done: {output}", file=sys.stderr)
        return output
    finally:
        for p in reversed(procs):
            _stop(p)
        for f in tmp_files:
            try:
                os.unlink(f)
            except OSError:
                pass
