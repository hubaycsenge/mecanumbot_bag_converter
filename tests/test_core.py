"""Pure-Python tests: no ROS graph, no bag files."""

import math
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

from bag_converter import colors, images, rviz_config
from bag_converter.displays import View, draw_points
from bag_converter.render import fit_view, up_angle_for
from bag_converter.tf_buffer import TfBuffer

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "model.rviz")


def _tf(parent, child, x=0.0, y=0.0, yaw=0.0, sec=0):
    return NS(
        header=NS(frame_id=parent, stamp=NS(sec=sec, nanosec=0)),
        child_frame_id=child,
        transform=NS(translation=NS(x=x, y=y, z=0.0),
                     rotation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))),
    )


# --- rviz config -------------------------------------------------------------------------

def test_rviz_config_reads_mecanumbot_model():
    cfg = rviz_config.load(FIXTURE)
    assert cfg.fixed_frame == "map"
    assert cfg.background == (48, 48, 48)
    topics = {d.topic for d in cfg.enabled_displays}
    assert {"/map", "/mecanumbot/scan", "/mecanumbot/people_fusion"} <= topics
    odom = next(d for d in cfg.displays if d.cls == "Odometry")
    assert not odom.enabled


def test_rviz_display_props():
    cfg = rviz_config.load(FIXTURE)
    scan = next(d for d in cfg.displays if d.cls == "LaserScan")
    assert scan.get("Color Transformer") == "Intensity"
    costmap = next(d for d in cfg.displays if d.topic == "/local_costmap/costmap")
    assert costmap.get("Color Scheme") == "costmap"
    assert costmap.nested("Update Topic", "Value") == "/local_costmap/costmap_updates"
    assert costmap.alpha() == pytest.approx(0.7)


def test_parse_color():
    assert rviz_config.parse_color("255; 25; 0") == (255, 25, 0)
    assert rviz_config.parse_color(None, (1, 2, 3)) == (1, 2, 3)
    assert rviz_config.parse_color("garbage", (1, 2, 3)) == (1, 2, 3)


def test_groups_are_flattened(tmp_path):
    p = tmp_path / "g.rviz"
    p.write_text(
        "Visualization Manager:\n"
        "  Displays:\n"
        "    - Class: rviz_common/Group\n"
        "      Enabled: true\n"
        "      Displays:\n"
        "        - Class: rviz_default_plugins/LaserScan\n"
        "          Enabled: true\n"
        "          Name: inner\n"
        "          Topic:\n"
        "            Value: /scan\n"
        "  Global Options:\n"
        "    Fixed Frame: odom\n")
    cfg = rviz_config.load(str(p))
    assert cfg.fixed_frame == "odom"
    assert [(d.cls, d.topic) for d in cfg.displays] == [("LaserScan", "/scan")]


# --- tf ------------------------------------------------------------------------------------

def test_tf_chain_and_inverse():
    tf = TfBuffer()
    tf.add(_tf("map", "odom", x=1.0), static=True)
    tf.add(_tf("odom", "base", x=2.0, yaw=math.pi / 2, sec=1))
    tf.add(_tf("base", "laser", x=0.5), static=True)
    t = tf.lookup("map", "laser", 1_000_000_000)
    # laser is 0.5 m along base's x, which points along map +y
    assert t[:2, 3] == pytest.approx([3.0, 0.5])
    back = tf.lookup("laser", "map", 1_000_000_000)
    assert (back @ t) == pytest.approx(np.eye(4))


def test_tf_interpolates_and_clamps():
    tf = TfBuffer()
    tf.add(_tf("map", "base", x=0.0, sec=1))
    tf.add(_tf("map", "base", x=2.0, sec=3))
    assert tf.lookup("map", "base", 2_000_000_000)[0, 3] == pytest.approx(1.0)
    assert tf.lookup("map", "base", 0)[0, 3] == pytest.approx(0.0)        # before: first
    assert tf.lookup("map", "base", 9_000_000_000)[0, 3] == pytest.approx(2.0)  # after: last


def test_tf_unknown_frame_and_leading_slash():
    tf = TfBuffer()
    tf.add(_tf("/map", "base", x=1.0), static=True)
    assert tf.lookup("map", "/base", 0)[0, 3] == pytest.approx(1.0)
    assert tf.lookup("map", "nowhere", 0) is None


def test_tf_clock_offset_is_median_receive_minus_stamp():
    tf = TfBuffer()
    for k in range(5):
        tf.add(_tf("map", "base", sec=k), receive_ns=k * 1_000_000_000 + 200_000_000)
    tf.finalize()
    assert tf.clock_offset_ns == 200_000_000


# --- view ------------------------------------------------------------------------------------

def test_north_view_puts_plus_y_up_and_plus_x_right():
    v = View(100, 100, (0, 0), 10, math.pi / 2)
    px = v.to_px(np.array([[0, 0], [1, 0], [0, 1]]))
    assert px[0] == pytest.approx([50, 50])
    assert px[1] == pytest.approx([60, 50])   # +x -> right
    assert px[2] == pytest.approx([50, 40])   # +y -> up (smaller row)


def test_rviz_orbit_yaw_zero_looks_down_minus_x():
    cfg = rviz_config.RvizConfig(path="", view=rviz_config.View(cls="Orbit", yaw=0.0))
    up = up_angle_for("rviz", cfg)
    v = View(100, 100, (0, 0), 10, up)
    # camera at +x looking back at the origin: -x is up on screen, +y to the right
    assert v.to_px(np.array([[-1, 0]]))[0] == pytest.approx([50, 40])
    assert v.to_px(np.array([[0, 1]]))[0] == pytest.approx([60, 50])


def test_fit_view_contains_all_points():
    pts = np.array([[-3, -1], [7, 4], [2, 9]])
    v = fit_view(pts, 640, 360, math.pi / 2)
    px = v.to_px(pts)
    assert (px[:, 0] >= 0).all() and (px[:, 0] <= 640).all()
    assert (px[:, 1] >= 0).all() and (px[:, 1] <= 360).all()


def test_draw_points_clips_to_image():
    img = np.zeros((10, 10, 3), np.uint8)
    draw_points(img, np.array([[0, 0], [9, 9], [50, 50]]), (1, 2, 3), 3)
    assert tuple(img[0, 0]) == (1, 2, 3) and tuple(img[9, 9]) == (1, 2, 3)


# --- colors ---------------------------------------------------------------------------------

def test_map_palette_matches_rviz():
    p = colors.map_palette()
    assert tuple(p[0]) == (255, 255, 255, 255)   # free
    assert tuple(p[100]) == (0, 0, 0, 255)       # occupied
    assert tuple(p[255][:3]) == colors.UNKNOWN_RGB   # -1 as uint8


def test_costmap_palette_free_is_transparent():
    p = colors.costmap_palette()
    assert p[0][3] == 0
    assert tuple(p[100][:3]) == (255, 0, 255)


def test_rainbow_endpoints():
    rgb = colors.rainbow([0.0, 1.0])
    assert rgb[0] == pytest.approx([1, 0, 1])   # value 0 -> magenta
    assert rgb[1] == pytest.approx([1, 0, 0])   # value 1 -> red


# --- images ---------------------------------------------------------------------------------

def _raw(encoding, arr, bigendian=False):
    arr = np.ascontiguousarray(arr)
    if bigendian:
        arr = arr.astype(arr.dtype.newbyteorder(">"))
    h, w = arr.shape[:2]
    return NS(encoding=encoding, height=h, width=w, step=arr.strides[0], data=arr.tobytes(),
              is_bigendian=bigendian)


def test_raw_rgb8_becomes_bgr():
    arr = np.zeros((2, 3, 3), np.uint8)
    arr[..., 0] = 255   # red in RGB
    out = images.to_bgr_or_gray(_raw("rgb8", arr))
    assert tuple(out[0, 0]) == (0, 0, 255)


def test_raw_mono16_bigendian_and_row_padding():
    arr = np.arange(6, dtype=np.uint16).reshape(2, 3) * 1000
    msg = _raw("mono16", arr, bigendian=True)
    assert (images.raw_to_array(msg) == arr).all()
    padded = np.zeros((2, 4), np.uint8)
    padded[:, :3] = [[1, 2, 3], [4, 5, 6]]
    msg = NS(encoding="mono8", height=2, width=3, step=4, data=padded.tobytes(),
             is_bigendian=False)
    assert images.raw_to_array(msg).tolist() == [[1, 2, 3], [4, 5, 6]]


def test_unsupported_encoding_raises():
    with pytest.raises(ValueError):
        images.raw_to_array(NS(encoding="weird", height=1, width=1, step=1, data=b"\0",
                               is_bigendian=False))


def test_compressed_kind_detects_payloads():
    assert images.compressed_kind(NS(format="jpeg", data=b"\xff\xd8\xff\xe0xx"))[0] == ".jpg"
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 8
    assert images.compressed_kind(NS(format="rgb8; png compressed", data=png))[0] == ".png"
    ext, payload = images.compressed_kind(
        NS(format="16UC1; compressedDepth png", data=b"\0" * 12 + png))
    assert ext == ".png" and payload == png


def test_resample_keeps_real_timing():
    times = [0, 100_000_000, 1_000_000_000]   # 0 s, 0.1 s, 1.0 s
    idx = images.resample_times(times, fps=10)
    assert len(idx) == 11
    assert idx[0] == 0 and idx[1] == 1 and idx[5] == 1 and idx[-1] == 2


def test_video_timeline_prefers_monotonic_stamps():
    frames = [(10, 1, b""), (11, 2, b""), (40, 3, b"")]
    times, clock = images.video_timeline(frames)
    assert clock == "header stamp" and list(times) == [1, 2, 3]
    times, clock = images.video_timeline([(10, 2, b""), (11, 1, b"")])
    assert clock == "bag receive time"
    times, clock = images.video_timeline([(10, 0, b""), (11, 1, b"")])
    assert clock == "bag receive time"


def test_estimate_fps_floor():
    assert images.estimate_fps([0, 500_000_000, 1_000_000_000]) == 30
    assert images.estimate_fps(list(range(0, 1_000_000_000, 10_000_000))) == 60


def test_sanitize_topic():
    assert images.sanitize("/a/b/compressed") == "a__b__compressed"
