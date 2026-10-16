"""
Export the image streams in a bag (sensor_msgs/Image and CompressedImage).

Decoding is done here rather than with cv_bridge: the Humble cv_bridge on
this machine is built against NumPy 1.x and fails to import under NumPy 2.
"""

import csv
import math
import os
import sys

import numpy as np

from . import bag_io

IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")

_CHANNELS = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1), "8UC3": (np.uint8, 3), "8UC4": (np.uint8, 4),
    "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1), "16SC1": (np.int16, 1),
    "rgb16": (np.uint16, 3), "bgr16": (np.uint16, 3), "rgba16": (np.uint16, 4),
    "bgra16": (np.uint16, 4), "32FC1": (np.float32, 1), "32SC1": (np.int32, 1),
    "64FC1": (np.float64, 1),
    "bayer_rggb8": (np.uint8, 1), "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1), "bayer_grbg8": (np.uint8, 1),
}


def image_topics(types):
    return sorted(t for t, ty in types.items() if ty in IMAGE_TYPES)


def sanitize(topic):
    return topic.strip("/").replace("/", "__") or "root"


def raw_to_array(msg):
    """Return a numpy array of a sensor_msgs/Image, in its native channel order."""
    enc = msg.encoding
    h, w = msg.height, msg.width
    buf = bytes(msg.data)
    if enc in ("yuv422", "uyvy", "yuv422_yuy2", "yuyv"):
        arr = np.frombuffer(buf, np.uint8).reshape(h, msg.step)[:, : w * 2].reshape(h, w, 2)
        return arr
    if enc not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {enc!r}")
    dtype, ch = _CHANNELS[enc]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_items = msg.step // dtype.itemsize
    arr = np.frombuffer(buf, dtype).reshape(h, row_items)[:, : w * ch]
    arr = arr.reshape(h, w, ch) if ch > 1 else arr.reshape(h, w)
    return arr.astype(arr.dtype.newbyteorder("="))


def to_bgr_or_gray(msg_or_arr, encoding=None):
    """Convert a raw image to something cv2.imwrite stores faithfully (BGR, BGRA or gray)."""
    import cv2

    if encoding is None:
        encoding = msg_or_arr.encoding
        arr = raw_to_array(msg_or_arr)
    else:
        arr = msg_or_arr
    if encoding in ("rgb8", "rgb16"):
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if encoding in ("rgba8", "rgba16"):
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    if encoding in ("uyvy", "yuv422"):
        return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_UYVY)
    if encoding in ("yuyv", "yuv422_yuy2"):
        return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUYV)
    if encoding.startswith("bayer_"):
        code = {
            "bayer_rggb8": cv2.COLOR_BayerRG2BGR, "bayer_bggr8": cv2.COLOR_BayerBG2BGR,
            "bayer_gbrg8": cv2.COLOR_BayerGB2BGR, "bayer_grbg8": cv2.COLOR_BayerGR2BGR,
        }[encoding]
        return cv2.cvtColor(arr, code)
    # bgr*, mono* and depth encodings are already in a layout cv2 stores as-is
    # (float and 32-bit data go to TIFF in _write_image).
    return arr


def compressed_kind(msg):
    """Return (extension, payload bytes) of a CompressedImage without re-encoding it."""
    data = bytes(msg.data)
    fmt = (msg.format or "").lower()
    if "compresseddepth" in fmt:
        # compressed_depth_image_transport prepends a 12-byte ConfigHeader.
        idx = data.find(b"\x89PNG")
        return ".png", data[idx:] if idx >= 0 else data[12:]
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg", data
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png", data
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", data
    if "png" in fmt:
        return ".png", data
    return ".jpg", data


def decode_compressed(msg):
    import cv2

    _, payload = compressed_kind(msg)
    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode compressed image")
    return img


def to_display_bgr(img):
    """Return an 8-bit 3-channel version of any decoded image, for video."""
    import cv2

    if img.dtype != np.uint8:
        f = img.astype(np.float64)
        finite = np.isfinite(f)
        if finite.any():
            lo, hi = np.percentile(f[finite], (1, 99))
            f = np.clip((f - lo) / max(hi - lo, 1e-9), 0, 1)
        f[~finite] = 0
        img = (f * 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _write_image(path_no_ext, img, fmt):
    import cv2

    if img.dtype in (np.float32, np.float64, np.int32):
        path = path_no_ext + ".tiff"
        cv2.imwrite(path, img.astype(np.float32))
        return path
    if fmt == "jpg" and img.dtype == np.uint8:
        path = path_no_ext + ".jpg"
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        path = path_no_ext + ".png"
        cv2.imwrite(path, img)
    return path


def resample_times(times_ns, fps, start_ns=None, end_ns=None):
    """
    Map constant-rate video ticks to source frame indices.

    Return, for each output tick, the index of the latest source frame at or
    before it, so the video keeps the recording's real timing (frames are
    repeated across gaps and dropped above the output rate).
    """
    times = np.asarray(times_ns, dtype=np.int64)
    if times.size == 0:
        return np.zeros(0, dtype=np.int64)
    start = times[0] if start_ns is None else start_ns
    end = times[-1] if end_ns is None else end_ns
    step = 1e9 / fps
    n = int(np.ceil((end - start) / step)) + 1   # last tick at/after the last frame
    ticks = start + (np.arange(n) * step).astype(np.int64)
    idx = np.searchsorted(times, ticks, side="right") - 1
    return np.clip(idx, 0, times.size - 1)


def estimate_fps(times_ns, floor=30.0, ceiling=60.0):
    """
    Return the output rate for a topic's video: 30 fps unless the topic is faster.

    The video is a real-time timeline, so frames are repeated across gaps; a
    rate below 30 would drop frames whenever the camera topic arrives in
    bursts (as it does when recorded over Wi-Fi), even if its mean rate is low.
    """
    if len(times_ns) < 2:
        return floor
    span = (int(times_ns[-1]) - int(times_ns[0])) / 1e9
    if span <= 0:
        return floor
    return float(np.clip(math.ceil((len(times_ns) - 1) / span), floor, ceiling))


def video_timeline(frames):
    """
    Pick the clock a topic's video is timed by: header stamps, else receive time.

    Header stamps are the camera's capture times and stay evenly spaced; bag
    receive times bunch up when the stream crosses Wi-Fi, which would make the
    video stutter and drop frames. Stamps are used only when every frame has
    one and they never go backwards. ``frames`` holds (receive_ns, stamp_ns, ...).
    """
    stamps = np.array([f[1] for f in frames], dtype=np.int64)
    if stamps.size and (stamps > 0).all() and (np.diff(stamps) >= 0).all():
        return stamps, "header stamp"
    return np.array([f[0] for f in frames], dtype=np.int64), "bag receive time"


def export_images(bag, out_dir, topics=None, fmt="original", frames=True, video=False,
                  video_fps=None, start_s=None, end_s=None, every=1, use_header_stamp=False):
    """
    Export each image topic to ``<out_dir>/<topic>/`` (frames + index.csv) and/or an .mp4.

    Return {topic: summary dict}.
    """
    from .video import VideoWriter

    bag = bag_io.resolve_bag(bag)
    types = bag_io.topic_types(bag)
    available = image_topics(types)
    if topics:
        missing = [t for t in topics if t not in available]
        if missing:
            raise SystemExit(f"not image topics in this bag: {missing}; available: {available}")
    else:
        topics = available
    if not topics:
        raise SystemExit("no sensor_msgs/Image or CompressedImage topics in this bag")

    bag_start, _ = bag_io.time_range(bag)
    lo = bag_start + int(start_s * 1e9) if start_s is not None else None
    hi = bag_start + int(end_s * 1e9) if end_s is not None else None

    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for topic in topics:
        tdir = os.path.join(out_dir, sanitize(topic))
        if frames:
            os.makedirs(tdir, exist_ok=True)
        state[topic] = {"dir": tdir, "seen": 0, "written": 0, "rows": [],
                        "video_frames": [], "errors": 0}

    for topic, msg, t_ns in bag_io.iter_messages(bag, topics, types):
        if lo is not None and t_ns < lo:
            continue
        if hi is not None and t_ns > hi:
            break
        st = state[topic]
        st["seen"] += 1
        if (st["seen"] - 1) % max(1, every):
            continue
        stamp = bag_io.stamp_ns(msg)
        key_time = stamp if (use_header_stamp and stamp) else t_ns
        i = st["written"]
        compressed = types[topic].endswith("CompressedImage")
        try:
            img = None
            if frames:
                base = os.path.join(st["dir"], f"frame_{i:06d}_{key_time}")
                if compressed and fmt == "original":
                    ext, payload = compressed_kind(msg)
                    path = base + ext
                    with open(path, "wb") as fh:
                        fh.write(payload)
                else:
                    img = decode_compressed(msg) if compressed else to_bgr_or_gray(msg)
                    path = _write_image(base, img, "jpg" if fmt == "jpg" else "png")
            else:
                path = ""
            if video:
                if compressed:
                    buf = compressed_kind(msg)[1]   # already compressed: keep as-is
                else:
                    if img is None:
                        img = to_bgr_or_gray(msg)
                    buf = _encode_for_buffer(img)
                st["video_frames"].append((t_ns, stamp, buf))
        except ValueError as exc:
            st["errors"] += 1
            if st["errors"] <= 3:
                print(f"warning: {topic} message {st['seen']}: {exc}", file=sys.stderr)
            continue
        h = getattr(msg, "height", "")   # CompressedImage carries no size;
        w = getattr(msg, "width", "")    # it is known only if we decoded it
        if img is not None:
            h, w = img.shape[:2]
        enc = getattr(msg, "encoding", None) or getattr(msg, "format", "")
        st["rows"].append([i, os.path.basename(path), t_ns, stamp,
                           msg.header.frame_id, w, h, enc])
        st["written"] += 1

    summary = {}
    for topic, st in state.items():
        info = {"frames": st["written"], "errors": st["errors"]}
        if frames and st["rows"]:
            index = os.path.join(st["dir"], "index.csv")
            with open(index, "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                wr.writerow(["index", "file", "bag_time_ns", "header_stamp_ns",
                             "frame_id", "width", "height", "encoding"])
                wr.writerows(st["rows"])
            info["dir"] = st["dir"]
        if video and st["video_frames"]:
            times, clock = video_timeline(st["video_frames"])
            fps = video_fps or estimate_fps(times)
            first = _decode_buffer(st["video_frames"][0][2])
            path = os.path.join(out_dir, sanitize(topic) + ".mp4")
            idx = resample_times(times, fps)
            with VideoWriter(path, first.shape[1], first.shape[0], fps) as vw:
                last_i, last_img = None, None
                for j in idx:
                    if j != last_i:
                        last_img = _decode_buffer(st["video_frames"][j][2])
                        last_i = j
                    vw.write(last_img)
            info["video"] = path
            info["video_fps"] = fps
            info["video_clock"] = clock
            info["video_dropped"] = len(times) - len(np.unique(idx))
        summary[topic] = info
    return summary


def _encode_for_buffer(img):
    """Keep raw video frames compressed in memory (JPEG q95) so long bags fit in RAM."""
    import cv2

    ok, buf = cv2.imencode(".jpg", to_display_bgr(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


def _decode_buffer(buf):
    import cv2

    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode buffered video frame")
    return to_display_bgr(img)
