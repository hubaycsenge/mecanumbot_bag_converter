"""
Top-down re-implementations of the RViz displays MecanumBot's model.rviz uses.

Each display keeps the latest state of its topic (as RViz does: a pose
stays on screen until the next one arrives; scans and clouds fade after
their Decay Time) and draws it onto a BGR canvas through a ``View``.
Everything is drawn in the fixed frame, ignoring height (z).
"""

import collections
import math
import sys

import cv2
import numpy as np

from . import bag_io
from .colors import bgr, palette, scalar_colors
from .geometry import heading_xy, pose_to_mat, transform_points

# Wheel/body sizes for the RobotModel stand-in (the URDF meshes are not drawn).
ROBOT_BODY = (-0.11, 0.17, -0.13, 0.13)   # x_min, x_max, y_min, y_max in base_link [m]
WHEEL_SIZE = (0.10, 0.05)                 # length, width [m]


class View:
    """Orthographic top-down camera: world (fixed frame) metres -> image pixels."""

    def __init__(self, width, height, center, scale, up_angle):
        self.width, self.height = int(width), int(height)
        self.center = np.asarray(center, dtype=float)
        self.scale = float(scale)          # pixels per metre
        self.up_angle = float(up_angle)    # world direction that points up on screen
        u = np.array([math.cos(up_angle), math.sin(up_angle)])
        r = np.array([u[1], -u[0]])
        s = self.scale
        # px = W/2 + s*r.(p-c) ; py = H/2 - s*u.(p-c)
        self.matrix = np.array([
            [s * r[0], s * r[1], self.width / 2 - s * r.dot(self.center)],
            [-s * u[0], -s * u[1], self.height / 2 + s * u.dot(self.center)],
        ])

    def to_px(self, pts):
        pts = np.asarray(pts, dtype=float)
        pts = pts.reshape(-1, pts.shape[-1])
        return pts[:, :2] @ self.matrix[:, :2].T + self.matrix[:, 2]

    def screen_angle(self, world_yaw):
        """Return the image-plane angle (radians, y down) of a world heading."""
        d = np.array([math.cos(world_yaw), math.sin(world_yaw)])
        v = self.matrix[:, :2] @ d
        return math.atan2(v[1], v[0])

    def key(self):
        return tuple(np.round(self.matrix.ravel(), 6))


def blend(img, alpha, draw):
    """Run ``draw(target)`` and composite it onto ``img`` at the given opacity."""
    if alpha >= 0.999:
        draw(img)
        return
    if alpha <= 0.001:
        return
    overlay = img.copy()
    draw(overlay)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def draw_points(img, px, colors, size_px):
    """Draw square points (vectorised); ``colors`` is (N, 3) BGR or a single BGR tuple."""
    if len(px) == 0:
        return
    h, w = img.shape[:2]
    pts = np.round(px).astype(np.int64)
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.ndim == 1:
        colors = np.broadcast_to(colors, (len(pts), 3))
    half = max(0, int(round(size_px)) // 2)
    offsets = range(-half, half + 1)
    for dx in offsets:
        for dy in offsets:
            x, y = pts[:, 0] + dx, pts[:, 1] + dy
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            img[y[ok], x[ok]] = colors[ok]


def draw_arrow(img, view, x, y, yaw, length, color, thickness, head_frac=0.3):
    start = view.to_px(np.array([[x, y]]))[0]
    end = view.to_px(np.array([[x + length * math.cos(yaw), y + length * math.sin(yaw)]]))[0]
    if np.linalg.norm(end - start) < 2:
        cv2.circle(img, tuple(np.round(start).astype(int)), max(2, thickness), color, -1,
                   cv2.LINE_AA)
        return
    cv2.arrowedLine(img, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)),
                    color, max(1, thickness), cv2.LINE_AA, tipLength=head_frac)


def draw_axes(img, view, t, length, thickness):
    o = t[:2, 3]
    for axis, color in ((0, (0, 0, 255)), (1, (0, 255, 0))):
        tip = o + length * t[:2, axis]
        a, b = view.to_px(np.array([o, tip]))
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color,
                 max(1, thickness), cv2.LINE_AA)


class Context:
    """Shared render state: the TF buffer, the fixed frame and the clock mapping."""

    def __init__(self, tf, fixed_frame, min_point_px=2.0, robot_frame=None):
        self.tf = tf
        self.fixed = fixed_frame
        self.min_point_px = min_point_px
        self.robot_frame = robot_frame
        self._warned = set()

    def stamp_for(self, msg, receive_ns):
        stamp = bag_io.stamp_ns(msg)
        return stamp if stamp else receive_ns - self.tf.clock_offset_ns

    def to_fixed(self, frame, stamp_ns, what):
        if not frame:
            frame = self.fixed
        t = self.tf.lookup(self.fixed, frame, stamp_ns)
        if t is None:
            self.warn((what, frame), f"{what}: no transform {frame} -> {self.fixed}; skipped")
        return t

    def warn(self, key, text):
        if key not in self._warned:
            self._warned.add(key)
            print(f"warning: {text}", file=sys.stderr)


class DisplayBase:
    layer = 50
    message_types = ()

    def __init__(self, display, ctx):
        self.d = display
        self.ctx = ctx
        self.topic = display.topic
        self.alpha = display.alpha()

    @property
    def legend_color(self):
        return bgr(self.d.color())

    def topics(self):
        return [self.topic] if self.topic else []

    def consume(self, topic, msg, t_ns):
        raise NotImplementedError

    def draw(self, img, view, now_ns):
        raise NotImplementedError


class GridDisplay(DisplayBase):
    layer = 10

    def topics(self):
        return []

    def draw(self, img, view, now_ns):
        n = int(self.d.get("Plane Cell Count", 10))
        size = float(self.d.get("Cell Size", 1.0))
        ref = self.d.get("Reference Frame", "<Fixed Frame>")
        t = np.eye(4)
        if ref and ref != "<Fixed Frame>":
            stamp = now_ns - self.ctx.tf.clock_offset_ns
            t = self.ctx.tf.lookup(self.ctx.fixed, ref, stamp)
            if t is None:
                return
        off = self.d.get("Offset", {}) or {}
        ox, oy = float(off.get("X", 0)), float(off.get("Y", 0))
        half = n * size / 2
        lines = []
        for k in range(n + 1):
            c = -half + k * size
            lines.append(((c + ox, -half + oy), (c + ox, half + oy)))
            lines.append(((-half + ox, c + oy), (half + ox, c + oy)))
        pts = np.array(lines, dtype=float).reshape(-1, 2)
        pts3 = np.c_[pts, np.zeros(len(pts))]
        px = view.to_px(transform_points(t, pts3)).reshape(-1, 2, 2)
        color = bgr(self.d.color())

        def _draw(target):
            for a, b in px:
                cv2.line(target, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                         color, 1, cv2.LINE_AA)
        blend(img, self.alpha, _draw)


class MapDisplay(DisplayBase):
    layer = 0

    def __init__(self, display, ctx):
        super().__init__(display, ctx)
        self.palette = palette(display.get("Color Scheme", "map"))
        self.update_topic = display.nested("Update Topic", "Value")
        self.grid = None      # (h, w) uint8 cell values
        self.info = None
        self.t_fixed_grid = None
        self.version = 0
        self._cache_key = None
        self._cache = None

    @property
    def legend_color(self):
        return bgr(self.palette[100][:3]) if self.palette[100][3] else (200, 200, 200)

    def topics(self):
        out = super().topics()
        if self.update_topic:
            out.append(self.update_topic)
        return out

    def consume(self, topic, msg, t_ns):
        if topic == self.topic:
            info = msg.info
            self.grid = np.asarray(msg.data, dtype=np.int8).view(np.uint8).reshape(
                info.height, info.width).copy()
            self.info = info
            stamp = self.ctx.stamp_for(msg, t_ns)
            frame_t = self.ctx.to_fixed(msg.header.frame_id, stamp, self.d.name)
            self.t_fixed_grid = None if frame_t is None else frame_t @ pose_to_mat(info.origin)
            self.version += 1
        elif topic == self.update_topic and self.grid is not None:
            patch = np.asarray(msg.data, dtype=np.int8).view(np.uint8).reshape(
                msg.height, msg.width)
            y0, x0 = msg.y, msg.x
            h = min(msg.height, self.grid.shape[0] - y0)
            w = min(msg.width, self.grid.shape[1] - x0)
            if h > 0 and w > 0:
                self.grid[y0:y0 + h, x0:x0 + w] = patch[:h, :w]
                self.version += 1

    def corners(self):
        """Return the map's four corners in the fixed frame (for fitting the view)."""
        if self.grid is None or self.t_fixed_grid is None:
            return None
        res = self.info.resolution
        h, w = self.grid.shape
        local = np.array([[0, 0, 0], [w * res, 0, 0], [w * res, h * res, 0], [0, h * res, 0]])
        return transform_points(self.t_fixed_grid, local)[:, :2]

    def _layers(self, view):
        key = (self.version, view.key(), img_shape(view))
        if key == self._cache_key:
            return self._cache
        res = self.info.resolution
        t = self.t_fixed_grid
        # cell (col, row) centre -> fixed frame -> pixel, as a single 2x3 affine.
        cell_to_world = np.array([
            [t[0, 0] * res, t[0, 1] * res, t[0, 3] + 0.5 * res * (t[0, 0] + t[0, 1])],
            [t[1, 0] * res, t[1, 1] * res, t[1, 3] + 0.5 * res * (t[1, 0] + t[1, 1])],
            [0, 0, 1],
        ])
        m = view.matrix @ cell_to_world
        rgba = self.palette[self.grid]
        warped = cv2.warpAffine(rgba, m, (view.width, view.height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = warped[:, :, 3:4].astype(np.float32) * (self.alpha / 255.0)
        color = warped[:, :, 2::-1].astype(np.float32) * a   # RGB -> BGR, premultiplied
        self._cache_key, self._cache = key, (color, 1.0 - a, bool((a > 0).any()))
        return self._cache

    def draw(self, img, view, now_ns):
        if self.grid is None or self.t_fixed_grid is None:
            return
        color, inv_a, visible = self._layers(view)
        if visible:
            np.copyto(img, (img.astype(np.float32) * inv_a + color).astype(np.uint8))


def img_shape(view):
    return view.width, view.height


class _DecayingPoints(DisplayBase):
    """Shared logic of LaserScan and PointCloud2: point sets that fade after Decay Time."""

    layer = 30

    def __init__(self, display, ctx):
        super().__init__(display, ctx)
        self.decay_ns = int(float(display.get("Decay Time", 0) or 0) * 1e9)
        self.sets = collections.deque()
        style = display.get("Style", "Flat Squares")
        if style == "Points":
            self.size_px = float(display.get("Size (Pixels)", 3))
            self.size_m = None
        else:
            self.size_px = None
            self.size_m = float(display.get("Size (m)", 0.01))

    def _colors(self, xyz, channels):
        d = self.d
        transformer = d.get("Color Transformer", "FlatColor")
        if transformer == "Intensity" and "intensity" in channels:
            vals = channels["intensity"]
            if d.get("Autocompute Intensity Bounds", True):
                return scalar_colors(vals, d)
            return scalar_colors(vals, d, float(d.get("Min Intensity", 0)),
                                 float(d.get("Max Intensity", 4096)))
        if transformer == "RGB8" and "rgb" in channels:
            return channels["rgb"]
        if transformer == "AxisColor" and len(xyz):
            axis = {"X": 0, "Y": 1, "Z": 2}.get(d.get("Axis", "Z"), 2)
            vals = xyz[:, axis]
            auto = d.nested("Autocompute Value Bounds", "Value", True)
            if auto:
                return scalar_colors(vals, d)
            bounds = d.get("Autocompute Value Bounds", {}) or {}
            return scalar_colors(vals, d, float(bounds.get("Min Value", -10)),
                                 float(bounds.get("Max Value", 10)))
        return np.asarray(bgr(d.color()), dtype=np.uint8)

    def _push(self, t_ns, xyz_fixed, colors):
        self.sets.append((t_ns, xyz_fixed[:, :2], colors))
        if self.decay_ns <= 0:
            while len(self.sets) > 1:
                self.sets.popleft()

    def draw(self, img, view, now_ns):
        if self.decay_ns > 0:
            while self.sets and self.sets[0][0] < now_ns - self.decay_ns:
                self.sets.popleft()
        size = self.size_px if self.size_px is not None else self.size_m * view.scale
        size = max(size, self.ctx.min_point_px)

        def _draw(target):
            for _, pts, colors in self.sets:
                draw_points(target, view.to_px(pts), colors, size)
        blend(img, self.alpha, _draw)


class LaserScanDisplay(_DecayingPoints):
    def consume(self, topic, msg, t_ns):
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        ok = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        channels = {}
        if len(msg.intensities) == len(ranges):
            channels["intensity"] = np.asarray(msg.intensities, dtype=float)[ok]
        r, a = ranges[ok], angles[ok]
        xyz = np.c_[r * np.cos(a), r * np.sin(a), np.zeros(len(r))]
        t = self.ctx.to_fixed(msg.header.frame_id, self.ctx.stamp_for(msg, t_ns), self.d.name)
        if t is None:
            return
        self._push(t_ns, transform_points(t, xyz), self._colors(xyz, channels))


_PC2_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pointcloud2_to_arrays(msg, max_points=200_000):
    """Return (xyz (N,3) float, channels dict) from a sensor_msgs/PointCloud2."""
    endian = ">" if msg.is_bigendian else "<"
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype in _PC2_DTYPES and f.count >= 1:
            names.append(f.name)
            formats.append(endian + _PC2_DTYPES[f.datatype])
            offsets.append(f.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                      "itemsize": msg.point_step})
    n = msg.width * msg.height
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.row_step and msg.height > 1 and msg.row_step != msg.width * msg.point_step:
        raw = raw.reshape(msg.height, msg.row_step)[:, : msg.width * msg.point_step].ravel()
    arr = np.frombuffer(raw.tobytes(), dtype=dtype, count=n)
    if n > max_points:
        arr = arr[:: int(math.ceil(n / max_points))]
    xyz = np.stack([arr[c].astype(float) for c in ("x", "y", "z")], axis=1)
    ok = np.isfinite(xyz).all(axis=1)
    xyz = xyz[ok]
    channels = {}
    for name in ("rgb", "rgba"):
        if name in names:
            packed = np.ascontiguousarray(arr[name][ok]).view(np.uint32)
            channels["rgb"] = np.stack(
                [packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF], 1).astype(np.uint8)
            break
    if "intensity" in names:
        channels["intensity"] = arr["intensity"][ok].astype(float)
    return xyz, channels


class PointCloud2Display(_DecayingPoints):
    def consume(self, topic, msg, t_ns):
        if "x" not in {f.name for f in msg.fields}:
            return
        xyz, channels = pointcloud2_to_arrays(msg)
        t = self.ctx.to_fixed(msg.header.frame_id, self.ctx.stamp_for(msg, t_ns), self.d.name)
        if t is None:
            return
        self._push(t_ns, transform_points(t, xyz), self._colors(xyz, channels))


class PathDisplay(DisplayBase):
    layer = 40

    def __init__(self, display, ctx):
        super().__init__(display, ctx)
        self.pts = None

    def consume(self, topic, msg, t_ns):
        if not msg.poses:
            self.pts = None
            return
        t = self.ctx.to_fixed(msg.header.frame_id, self.ctx.stamp_for(msg, t_ns), self.d.name)
        if t is None:
            return
        xyz = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                        for p in msg.poses])
        self.pts = transform_points(t, xyz)[:, :2]

    def draw(self, img, view, now_ns):
        if self.pts is None or len(self.pts) < 2:
            return
        px = np.round(view.to_px(self.pts)).astype(np.int32)
        width = 1
        if self.d.get("Line Style") == "Billboards":
            width = max(1, int(round(float(self.d.get("Line Width", 0.03)) * view.scale)))
        color = bgr(self.d.color())
        blend(img, self.alpha,
              lambda target: cv2.polylines(target, [px], False, color, max(2, width), cv2.LINE_AA))


class _PoseLike(DisplayBase):
    layer = 60

    def _arrow_params(self, view):
        d = self.d
        shape = d.get("Shape", "Arrow")
        if shape in ("Arrow (Flat)",):
            length = float(d.get("Arrow Length", 0.3))
            thick = 2
            head = 0.3
        else:
            shaft = float(d.get("Shaft Length", 0.23))
            head_len = float(d.get("Head Length", 0.07))
            length = shaft + head_len
            # rviz passes the "Radius" properties to Arrow as diameters
            diameter = float(d.get("Shaft Radius", 0.01))
            thick = max(2, int(round(diameter * view.scale)))
            head = head_len / length if length > 0 else 0.3
        return shape, length, thick, head

    def _draw_poses(self, img, view, mats):
        if not mats:
            return
        shape, length, thick, head = self._arrow_params(view)
        color = bgr(self.d.color())
        axes_len = float(self.d.get("Axes Length", 0.3))

        def _draw(target):
            for t in mats:
                if shape == "Axes":
                    draw_axes(target, view, t, axes_len, 2)
                else:
                    draw_arrow(target, view, t[0, 3], t[1, 3], heading_xy(t), length, color,
                               thick, head)
        blend(img, self.alpha, _draw)


class PoseDisplay(_PoseLike):
    def __init__(self, display, ctx):
        super().__init__(display, ctx)
        self.mat = None

    def consume(self, topic, msg, t_ns):
        pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose   # WithCovariance too
        t = self.ctx.to_fixed(msg.header.frame_id, self.ctx.stamp_for(msg, t_ns), self.d.name)
        if t is not None:
            self.mat = t @ pose_to_mat(pose)

    def draw(self, img, view, now_ns):
        self._draw_poses(img, view, [self.mat] if self.mat is not None else [])


class PoseArrayDisplay(_PoseLike):
    def __init__(self, display, ctx):
        super().__init__(display, ctx)
        self.mats = []

    def consume(self, topic, msg, t_ns):
        t = self.ctx.to_fixed(msg.header.frame_id, self.ctx.stamp_for(msg, t_ns), self.d.name)
        if t is not None:
            self.mats = [t @ pose_to_mat(p) for p in msg.poses]

    def draw(self, img, view, now_ns):
        self._draw_poses(img, view, self.mats)


class OdometryDisplay(_PoseLike):
    def __init__(self, display, ctx):
        super().__init__(display, ctx)
        self.keep = int(display.get("Keep", 100) or 100)
        self.pos_tol = float(display.get("Position Tolerance", 0.1))
        self.ang_tol = float(display.get("Angle Tolerance", 0.1))
        self.mats = collections.deque(maxlen=max(1, self.keep))

    @property
    def legend_color(self):
        return bgr(parse_nested_color(self.d, "Shape", (255, 25, 0)))

    def consume(self, topic, msg, t_ns):
        t = self.ctx.to_fixed(msg.header.frame_id, self.ctx.stamp_for(msg, t_ns), self.d.name)
        if t is None:
            return
        m = t @ pose_to_mat(msg.pose.pose)
        if self.mats:
            last = self.mats[-1]
            moved = np.linalg.norm(m[:2, 3] - last[:2, 3])
            turned = abs((heading_xy(m) - heading_xy(last) + math.pi) % (2 * math.pi) - math.pi)
            if moved < self.pos_tol and turned < self.ang_tol:
                return
        self.mats.append(m)

    def draw(self, img, view, now_ns):
        shape = self.d.props.get("Shape", {}) or {}
        color = bgr(parse_nested_color(self.d, "Shape", (255, 25, 0)))
        length = float(shape.get("Shaft Length", 1)) + float(shape.get("Head Length", 0.3))
        alpha = float(shape.get("Alpha", 1))

        def _draw(target):
            for m in self.mats:
                draw_arrow(target, view, m[0, 3], m[1, 3], heading_xy(m), length, color, 2)
        blend(img, alpha, _draw)


def parse_nested_color(display, key, default):
    from .rviz_config import parse_color
    value = display.props.get(key)
    return parse_color(value.get("Color"), default) if isinstance(value, dict) else default


class RobotModelDisplay(DisplayBase):
    """
    Stand-in for RobotModel: the base footprint plus the base_link's TF children.

    The URDF meshes are not rendered; the body box and wheels are drawn at the
    poses TF gives them, so grippers, head and wheels move as recorded.
    """

    layer = 70

    @property
    def legend_color(self):
        return (200, 200, 200)

    def topics(self):
        return []

    def draw(self, img, view, now_ns):
        base = self.ctx.robot_frame
        if not base:
            return
        tf = self.ctx.tf
        stamp = now_ns - tf.clock_offset_ns
        t_base = tf.lookup(self.ctx.fixed, base, stamp)
        if t_base is None:
            self.ctx.warn(("robot", base), f"RobotModel: no transform {base} -> {self.ctx.fixed}")
            return
        x0, x1, y0, y1 = ROBOT_BODY
        body = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
        body_px = np.round(view.to_px(transform_points(t_base, body))).astype(np.int32)

        def _draw(target):
            cv2.fillPoly(target, [body_px], (90, 90, 90), cv2.LINE_AA)
            cv2.polylines(target, [body_px], True, (220, 220, 220), 2, cv2.LINE_AA)
            for child in tf.children_of(base):
                t_child = tf.lookup(self.ctx.fixed, child, stamp)
                if t_child is None:
                    continue
                if "wheel" in child:
                    lx, ly = WHEEL_SIZE[0] / 2, WHEEL_SIZE[1] / 2
                    rect = np.array([[-lx, -ly, 0], [lx, -ly, 0], [lx, ly, 0], [-lx, ly, 0]])
                    # wheel frames are rolled 90 deg; draw them aligned with the base instead
                    t_w = t_base.copy()
                    t_w[:3, 3] = t_child[:3, 3]
                    px = np.round(view.to_px(transform_points(t_w, rect))).astype(np.int32)
                    cv2.fillPoly(target, [px], (30, 30, 30), cv2.LINE_AA)
                else:
                    c = np.round(view.to_px(t_child[:2, 3][None]))[0].astype(int)
                    cv2.circle(target, tuple(c), max(3, int(0.025 * view.scale)), (0, 200, 255),
                               -1, cv2.LINE_AA)
            # heading tick
            draw_arrow(target, view, t_base[0, 3], t_base[1, 3], heading_xy(t_base),
                       x1 + 0.08, (255, 255, 255), 2, 0.35)
        blend(img, self.alpha, _draw)


class TFDisplay(DisplayBase):
    layer = 80

    def topics(self):
        return []

    def draw(self, img, view, now_ns):
        scale = float(self.d.get("Marker Scale", 1.0))
        stamp = now_ns - self.ctx.tf.clock_offset_ns
        for frame in sorted(self.ctx.tf.frames):
            t = self.ctx.tf.lookup(self.ctx.fixed, frame, stamp)
            if t is not None:
                draw_axes(img, view, t, 0.1 * scale, 1)


DISPLAY_CLASSES = {
    "Grid": GridDisplay,
    "Map": MapDisplay,
    "LaserScan": LaserScanDisplay,
    "PointCloud2": PointCloud2Display,
    "Path": PathDisplay,
    "Pose": PoseDisplay,
    "PoseWithCovariance": PoseDisplay,
    "PoseArray": PoseArrayDisplay,
    "Odometry": OdometryDisplay,
    "RobotModel": RobotModelDisplay,
    "TF": TFDisplay,
}
