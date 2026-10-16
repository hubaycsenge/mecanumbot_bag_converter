"""Command line: ``bag_convert {info,render,images,capture} BAG ...``."""

import argparse
import os
import sys

from . import bag_io


def _size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must look like 1920x1080") from exc
    return w, h


def _config(path):
    from .rviz_config import default_config_path

    path = path or default_config_path()
    if not path or not os.path.isfile(path):
        raise SystemExit("no .rviz config found; pass one with -c/--config")
    return path


def cmd_info(args):
    from .images import image_topics

    bag = bag_io.resolve_bag(args.bag)
    types = bag_io.topic_types(bag)
    counts = bag_io.topic_counts(bag)
    start, end = bag_io.time_range(bag)
    print(f"bag:      {bag}")
    print(f"storage:  {bag_io.storage_id(bag)}")
    print(f"duration: {(end - start) / 1e9:.1f} s")
    print("topics:")
    for topic in sorted(types):
        print(f"  {topic:60s} {types[topic]:45s} {counts.get(topic, '?')}")
    imgs = image_topics(types)
    print("image topics (for `images`): " + (", ".join(imgs) if imgs else "none"))
    from .displays import DISPLAY_CLASSES
    from .rviz_config import load

    cfg_path = _config(args.config)
    cfg = load(cfg_path)
    print(f"rviz config: {cfg_path} (fixed frame {cfg.fixed_frame})")
    for d in cfg.displays:
        if not d.enabled:
            state = "disabled"
        elif d.cls not in DISPLAY_CLASSES:
            state = "NOT SUPPORTED by render"
        elif d.topic and d.topic not in types and d.cls not in ("RobotModel",):
            state = "not in bag"
        else:
            state = "drawn"
        print(f"  {d.cls:14s} {str(d.topic or ''):50s} {state}")


def cmd_render(args):
    from .render import render_bag

    bag = bag_io.resolve_bag(args.bag)
    out = args.output or f"{bag_io.bag_name(bag)}_rviz.mp4"
    overrides = dict(o.split("=", 1) for o in args.remap or [])
    rotation = args.rotation
    if rotation == "auto":
        rotation = "rviz" if args.view == "rviz" else "north"
    render_bag(bag, out, _config(args.config), fps=args.fps, speed=args.speed, size=args.size,
               view_mode=args.view, rotation=rotation, follow_frame=args.robot_frame,
               view_size=args.view_size, start_s=args.start, end_s=args.end,
               min_point_px=args.min_point_px, hud=not args.no_hud, legend=not args.no_legend,
               topic_overrides=overrides, crf=args.crf)


def cmd_images(args):
    from .images import export_images

    bag = bag_io.resolve_bag(args.bag)
    out = args.output or f"{bag_io.bag_name(bag)}_images"
    summary = export_images(bag, out, topics=args.topic, fmt=args.format,
                            frames=not args.no_frames, video=args.video or args.no_frames,
                            video_fps=args.fps, start_s=args.start, end_s=args.end,
                            every=args.every, use_header_stamp=args.header_stamp)
    for topic, info in summary.items():
        parts = [f"{info['frames']} frames"]
        if info.get("dir"):
            parts.append(f"-> {info['dir']}/ (index.csv)")
        if info.get("video"):
            parts.append(f"video {info['video']} @ {info['video_fps']:g} fps "
                         f"(timed by {info['video_clock']})")
            if info.get("video_dropped"):
                parts.append(f"{info['video_dropped']} frames above that rate not in the "
                             "video (raise --fps to keep them)")
        if info["errors"]:
            parts.append(f"{info['errors']} undecodable")
        print(f"{topic}: " + ", ".join(parts))


def cmd_capture(args):
    from .capture import capture_bag

    bag = bag_io.resolve_bag(args.bag)
    out = os.path.abspath(args.output or f"{bag_io.bag_name(bag)}_rviz_capture.mp4")
    capture_bag(bag, out, _config(args.config), rate=args.rate, fps=args.fps,
                domain_id=args.domain_id, startup_wait=args.startup_wait, urdf=args.urdf,
                start_s=args.start, duration_s=args.duration,
                virtual_display=args.virtual_display, crf=args.crf)


def build_parser():
    p = argparse.ArgumentParser(
        prog="bag_convert",
        description="Convert MecanumBot ROS 2 bags: RViz view -> .mp4, image topics -> files.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, config=True):
        sp.add_argument("bag", help="bag directory (or its .db3/.mcap/metadata.yaml)")
        if config:
            sp.add_argument("-c", "--config", help="RViz config (default: mecanumbot_description "
                            "rviz/model.rviz)")

    sp = sub.add_parser("info", help="list topics and what each RViz display will show")
    common(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("render", help="render the RViz displays top-down into an .mp4 "
                        "(offline, headless, frame-exact)")
    common(sp)
    sp.add_argument("-o", "--output", help="output .mp4 (default: <bag>_rviz.mp4)")
    sp.add_argument("--fps", type=float, default=30.0)
    sp.add_argument("--speed", type=float, default=1.0, help="playback speed factor (2 = 2x)")
    sp.add_argument("--size", type=_size, default=(1920, 1080), help="WxH (default 1920x1080)")
    sp.add_argument("--view", choices=("fit", "follow", "rviz"), default="fit",
                    help="fit: whole map + trajectory; follow: centred on the robot; "
                    "rviz: the config's saved camera")
    sp.add_argument("--rotation", choices=("auto", "rviz", "north", "east"), default="auto",
                    help="screen orientation: rviz = the config's camera yaw; north = +y up; "
                    "auto = rviz for --view rviz, north otherwise")
    sp.add_argument("--view-size", type=float, default=8.0,
                    help="metres shown vertically in --view follow")
    sp.add_argument("--robot-frame", help="robot TF frame (default: *base_link)")
    sp.add_argument("--start", type=float, help="seconds from bag start")
    sp.add_argument("--end", type=float, help="seconds from bag start")
    sp.add_argument("--min-point-px", type=float, default=2.0,
                    help="smallest drawn scan/cloud point, pixels")
    sp.add_argument("--remap", action="append", metavar="FROM=TO",
                    help="draw a display's topic from another bag topic")
    sp.add_argument("--no-hud", action="store_true", help="no time/bag overlay")
    sp.add_argument("--no-legend", action="store_true")
    sp.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better)")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("images", help="export image topics as separate image files "
                        "(+ index.csv) and/or one .mp4 per topic")
    common(sp, config=False)
    sp.add_argument("-o", "--output", help="output directory (default: <bag>_images/)")
    sp.add_argument("-t", "--topic", action="append",
                    help="image topic (repeatable; default: every image topic)")
    sp.add_argument("--format", choices=("original", "png", "jpg"), default="original",
                    help="original keeps compressed payloads byte-for-byte; raw images "
                    "become png")
    sp.add_argument("--video", action="store_true", help="also write <topic>.mp4")
    sp.add_argument("--no-frames", action="store_true", help="only the .mp4, no image files")
    sp.add_argument("--fps", type=float, 
                    help="video fps (default: 30, or the topic's mean rate if higher)")
    sp.add_argument("--every", type=int, default=1, help="keep every Nth message")
    sp.add_argument("--start", type=float, help="seconds from bag start")
    sp.add_argument("--end", type=float, help="seconds from bag start")
    sp.add_argument("--header-stamp", action="store_true",
                    help="name/time frames by header stamp instead of bag receive time")
    sp.set_defaults(func=cmd_images)

    sp = sub.add_parser("capture", help="screen-record the real RViz2 during `ros2 bag play` "
                        "(real time, on a private Xvfb/Xephyr display)")
    common(sp)
    sp.add_argument("-o", "--output", help="output .mp4 (default: <bag>_rviz_capture.mp4)")
    sp.add_argument("--rate", type=float, default=1.0,
                    help="bag play rate; the video is retimed back to real time")
    sp.add_argument("--fps", type=int, default=30)
    sp.add_argument("--domain-id", type=int, default=77,
                    help="isolated ROS_DOMAIN_ID for the replay (never the robot's 19)")
    sp.add_argument("--startup-wait", type=float, default=8.0,
                    help="seconds to let RViz load before playing")
    sp.add_argument("--urdf", help="URDF for RobotModel (default: mecanumbot_description)")
    sp.add_argument("--start", type=float, help="seconds from bag start")
    sp.add_argument("--duration", type=float, help="seconds of bag to record")
    sp.add_argument("--display", dest="virtual_display",
                    choices=("auto", "xvfb", "xephyr", "desktop"), default="auto",
                    help="X server RViz runs on. auto = Xvfb, else Xephyr (private, only RViz "
                    "is recorded); desktop = your screen, which records whatever covers RViz")
    sp.add_argument("--crf", type=int, default=20)
    sp.set_defaults(func=cmd_capture)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) == "capture" and args.domain_id == 19:
        print("refusing --domain-id 19: that is the live robot's domain", file=sys.stderr)
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
