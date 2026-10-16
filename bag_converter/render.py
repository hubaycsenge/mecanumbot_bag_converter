"""
Render a bag as an RViz-style top-down .mp4, driven by an .rviz config.

Two passes over the bag: the first loads all of /tf and /tf_static (and the
map bounds, to fit the view); the second streams the displayed topics in
bag order and emits a frame each time bag time crosses the next frame tick.
"""

import datetime
import math
import sys

import cv2
import numpy as np

from . import bag_io
from .displays import DISPLAY_CLASSES, Context, MapDisplay, View
from .rviz_config import load as load_rviz
from .tf_buffer import TfBuffer
from .video import VideoWriter

RVIZ_FOV = 0.785398  # rviz Orbit/FPS vertical field of view [rad]


def load_tf(bag, types):
    tf = TfBuffer()
    topics = [t for t in ("/tf", "/tf_static") if t in types]
    for topic, msg, t_ns in bag_io.iter_messages(bag, topics, types):
        static = topic == "/tf_static"
        for tr in msg.transforms:
            tf.add(tr, static=static, receive_ns=t_ns)
    tf.finalize()
    return tf


def guess_robot_frame(tf, requested=None):
    if requested:
        return requested
    frames = sorted(tf.frames)
    for suffix in ("base_link", "base_footprint"):
        for f in frames:
            if f.split("/")[-1] == suffix:
                return f
    return None


def up_angle_for(rotation, cfg):
    """Return the world direction drawn as "up" on screen."""
    if rotation == "north":
        return math.pi / 2           # +y up, +x right: a plain map view
    if rotation == "east":
        return 0.0
    if cfg.view.cls in ("TopDownOrtho",):
        return cfg.view.yaw + math.pi / 2
    # Orbit / FPS: the camera sits at yaw around the focal point and looks back at it.
    return cfg.view.yaw + math.pi


def fit_view(points, width, height, up_angle, margin=0.06):
    pts = np.asarray(points, dtype=float)
    u = np.array([math.cos(up_angle), math.sin(up_angle)])
    r = np.array([u[1], -u[0]])
    sx, sy = pts @ r, pts @ u
    span_x = max(sx.max() - sx.min(), 1.0)
    span_y = max(sy.max() - sy.min(), 1.0)
    scale = min(width * (1 - 2 * margin) / span_x, height * (1 - 2 * margin) / span_y)
    cx, cy = (sx.max() + sx.min()) / 2, (sy.max() + sy.min()) / 2
    center = cx * r + cy * u
    return View(width, height, center, scale, up_angle)


def _robot_track(tf, fixed, robot, n=300):
    rng = None
    # use the first dynamic edge on the robot's chain to know the recorded time span
    frame = robot
    seen = set()
    while frame and frame not in seen:
        seen.add(frame)
        rng = tf.time_range(frame)
        if rng:
            break
        edge = tf._edges.get(frame)
        frame = edge.parent if edge else None
    if not rng:
        return np.zeros((0, 2))
    pts = []
    for t in np.linspace(rng[0], rng[1], n).astype(np.int64):
        m = tf.lookup(fixed, robot, int(t))
        if m is not None:
            pts.append(m[:2, 3])
    return np.asarray(pts).reshape(-1, 2)


def draw_hud(img, text_lines, legend, view, show_legend=True, show_scale=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, img.shape[0] / 1400)
    th = max(1, int(round(fs * 2)))
    y = int(30 * fs * 1.4)
    for line in text_lines:
        cv2.putText(img, line, (16, y), font, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(img, line, (16, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        y += int(32 * fs * 1.2)
    if show_legend and legend:
        lh = int(26 * fs * 1.2)
        y = img.shape[0] - 16 - lh * (len(legend) - 1)
        for color, label in legend:
            cv2.rectangle(img, (16, y - int(14 * fs)), (16 + int(22 * fs), y + int(4 * fs)),
                          color, -1)
            cv2.putText(img, label, (24 + int(22 * fs), y), font, fs * 0.75, (0, 0, 0), th + 2,
                        cv2.LINE_AA)
            cv2.putText(img, label, (24 + int(22 * fs), y), font, fs * 0.75, (235, 235, 235),
                        max(1, th - 1), cv2.LINE_AA)
            y += lh
    if show_scale:
        target = 120.0 / view.scale   # metres covered by ~120 px
        nice = min((1, 2, 5, 10, 20, 50, 0.5, 0.2, 0.1), key=lambda m: abs(math.log(m / target)))
        length = int(round(nice * view.scale))
        x1 = img.shape[1] - 24
        x0 = x1 - length
        yb = img.shape[0] - 24
        cv2.line(img, (x0, yb), (x1, yb), (255, 255, 255), 3)
        for x in (x0, x1):
            cv2.line(img, (x, yb - 8), (x, yb + 2), (255, 255, 255), 3)
        label = f"{nice:g} m"
        cv2.putText(img, label, (x0, yb - 14), font, fs * 0.75, (255, 255, 255), max(1, th - 1),
                    cv2.LINE_AA)


def render_bag(bag, output, config_path, fps=30.0, speed=1.0, size=(1920, 1080),
               view_mode="fit", rotation="rviz", follow_frame=None, view_size=8.0,
               start_s=None, end_s=None, min_point_px=2.0, hud=True, legend=True,
               topic_overrides=None, crf=20):
    bag = bag_io.resolve_bag(bag)
    cfg = load_rviz(config_path)
    types = bag_io.topic_types(bag)
    width, height = size

    print(f"[1/2] loading TF from {bag_io.bag_name(bag)} ...", file=sys.stderr)
    tf = load_tf(bag, types)
    robot = guess_robot_frame(tf, follow_frame)
    ctx = Context(tf, cfg.fixed_frame, min_point_px=min_point_px, robot_frame=robot)
    if tf.clock_offset_ns and abs(tf.clock_offset_ns) > 50_000_000:
        print(f"      robot clock offset vs recorder: {tf.clock_offset_ns / 1e9:+.3f} s",
              file=sys.stderr)

    displays, skipped = [], []
    for d in cfg.enabled_displays:
        if topic_overrides and d.topic in topic_overrides:
            d.topic = topic_overrides[d.topic]
        cls = DISPLAY_CLASSES.get(d.cls)
        if cls is None:
            skipped.append(f"{d.cls} ({d.topic or d.name})")
            continue
        disp = cls(d, ctx)
        wanted = disp.topics()
        if wanted and not any(t in types for t in wanted):
            skipped.append(f"{d.cls} {d.topic} (not in bag)")
            continue
        displays.append(disp)
    displays.sort(key=lambda x: x.layer)
    if skipped:
        print("      not drawn: " + "; ".join(skipped), file=sys.stderr)

    routes = {}
    for disp in displays:
        for t in disp.topics():
            if t in types:
                routes.setdefault(t, []).append(disp)

    bag_start, bag_end = bag_io.time_range(bag)
    t0 = bag_start + int((start_s or 0) * 1e9)
    t1 = bag_start + int(end_s * 1e9) if end_s is not None else bag_end
    t1 = min(t1, bag_end)
    step_ns = int(1e9 * speed / fps)
    n_frames = max(1, int((t1 - t0) // step_ns) + 1)

    up = up_angle_for(rotation, cfg)
    static_view = None
    if view_mode == "fit":
        pts = []
        map_topics = [d.topic for d in displays if isinstance(d, MapDisplay) and d.topic in types]
        probes = [d for d in displays if isinstance(d, MapDisplay)]
        for topic, msg, t_ns in bag_io.iter_messages(bag, map_topics[:1], types):
            for p in probes:
                if p.topic == topic:
                    p.consume(topic, msg, t_ns)
                    c = p.corners()
                    if c is not None:
                        pts.extend(c)
            break
        for p in probes:   # reset; the main pass replays the maps in order
            p.grid = p.t_fixed_grid = None
        if robot:
            pts.extend(_robot_track(tf, cfg.fixed_frame, robot))
        if not pts:
            pts = [(-5, -5), (5, 5)]
        static_view = fit_view(np.asarray(pts), width, height, up)
    elif view_mode == "rviz":
        visible_h = 2 * cfg.view.distance * math.tan(RVIZ_FOV / 2)
        static_view = View(width, height, (cfg.view.focal_x, cfg.view.focal_y),
                           height / visible_h, up)
    elif view_mode != "follow":
        raise ValueError(f"unknown view mode {view_mode!r}")
    if view_mode == "follow" and not robot:
        raise SystemExit("--view follow needs a robot frame; none found in TF (use --robot-frame)")

    legend_items = []
    for disp in displays:
        label = disp.d.topic or disp.d.name
        if disp.d.cls in ("Grid",):
            continue
        legend_items.append((disp.legend_color, f"{disp.d.cls}: {label}"))

    background = np.empty((height - height % 2, width - width % 2, 3), np.uint8)
    background[:] = cfg.background[::-1]
    name = bag_io.bag_name(bag)

    def frame_view(now_ns):
        if static_view is not None:
            return static_view
        m = tf.lookup(cfg.fixed_frame, robot, now_ns - tf.clock_offset_ns)
        center = m[:2, 3] if m is not None else (0.0, 0.0)
        return View(background.shape[1], background.shape[0], center,
                    background.shape[0] / view_size, up)

    def render(now_ns):
        img = background.copy()
        view = frame_view(now_ns)
        for disp in displays:
            disp.draw(img, view, now_ns)
        if hud:
            wall = datetime.datetime.fromtimestamp(now_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f")
            lines = [name, f"t = {(now_ns - bag_start) / 1e9:7.2f} s   {wall[:-3]}"]
            if speed != 1.0:
                lines.append(f"{speed:g}x")
            draw_hud(img, lines, legend_items, view, show_legend=legend)
        return img

    print(f"[2/2] rendering {n_frames} frames at {fps:g} fps "
          f"({(t1 - t0) / 1e9:.1f} s of bag, {speed:g}x) -> {output}", file=sys.stderr)
    k = 0
    next_tick = t0
    report = max(1, n_frames // 20)
    with VideoWriter(output, background.shape[1], background.shape[0], fps, crf=crf) as vw:
        for topic, msg, t_ns in bag_io.iter_messages(bag, list(routes), types):
            while t_ns > next_tick and k < n_frames:
                vw.write(render(next_tick))
                k += 1
                next_tick = t0 + k * step_ns
                if k % report == 0:
                    print(f"      {100 * k // n_frames:3d}%  frame {k}/{n_frames}",
                          file=sys.stderr)
            if k >= n_frames:
                break
            for disp in routes[topic]:
                disp.consume(topic, msg, t_ns)
        while k < n_frames:
            vw.write(render(next_tick))
            k += 1
            next_tick = t0 + k * step_ns
    print(f"done: {output} ({k} frames, {k / fps:.1f} s)", file=sys.stderr)
    return output
