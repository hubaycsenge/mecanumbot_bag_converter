"""
Thin wrapper over rosbag2_py.

The ROS imports are deferred so the pure-Python parts of the package (and
its tests) work without a sourced ROS 2 environment.
"""

import os
import sys

import yaml


def resolve_bag(path):
    """Accept a bag directory, its metadata.yaml or one of its storage files."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        parent = os.path.dirname(path)
        if os.path.isfile(os.path.join(parent, "metadata.yaml")):
            return parent
    return path


def read_metadata(bag):
    meta_path = os.path.join(bag, "metadata.yaml")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("rosbag2_bagfile_information", {}) or {}


def storage_id(bag):
    sid = read_metadata(bag).get("storage_identifier")
    if sid:
        return sid
    if os.path.isfile(bag):
        return "mcap" if bag.endswith(".mcap") else "sqlite3"
    for name in os.listdir(bag):
        if name.endswith(".mcap"):
            return "mcap"
    return "sqlite3"


def bag_name(bag):
    return os.path.basename(os.path.normpath(bag))


def _open(bag, topics=None):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id=storage_id(bag)),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


def topic_types(bag):
    """Return {topic: type} for every topic in the bag."""
    reader = _open(bag)
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def topic_counts(bag):
    """Return {topic: message_count} from metadata.yaml (empty if unavailable)."""
    out = {}
    for entry in read_metadata(bag).get("topics_with_message_count", []) or []:
        out[entry["topic_metadata"]["name"]] = entry.get("message_count", 0)
    return out


def time_range(bag):
    """Return (start_ns, end_ns) of the bag from its metadata."""
    meta = read_metadata(bag)
    start = int((meta.get("starting_time") or {}).get("nanoseconds_since_epoch", 0))
    dur = int((meta.get("duration") or {}).get("nanoseconds", 0))
    return start, start + dur


def message_classes(types, topics):
    """Import message classes, skipping (with a warning) the ones that cannot be imported."""
    from rosidl_runtime_py.utilities import get_message

    out = {}
    for topic in topics:
        type_name = types.get(topic)
        if type_name is None:
            continue
        try:
            out[topic] = get_message(type_name)
        except (AttributeError, ModuleNotFoundError, ValueError) as exc:
            print(f"warning: cannot import {type_name} for {topic} ({exc}); "
                  "is the workspace sourced?", file=sys.stderr)
    return out


def iter_messages(bag, topics, types=None):
    """Yield (topic, message, receive_time_ns) in bag order for the given topics."""
    from rclpy.serialization import deserialize_message

    types = types or topic_types(bag)
    classes = message_classes(types, topics)
    if not classes:
        return
    reader = _open(bag, classes.keys())
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        cls = classes.get(topic)
        if cls is not None:
            yield topic, deserialize_message(data, cls), t_ns


def stamp_ns(msg, fallback=0):
    header = getattr(msg, "header", None)
    if header is None:
        return fallback
    ns = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
    return ns or fallback
