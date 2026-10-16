"""Parse an RViz2 ``.rviz`` file into the displays the offline renderer draws."""

import os
from dataclasses import dataclass, field

import yaml

DEFAULT_CONFIG_CANDIDATES = (
    os.path.expanduser(
        "~/Documents/mecanumbot_ws/src/mecanumbot/mecanumbot_description/rviz/model.rviz"),
)


@dataclass
class Display:
    cls: str           # short class name, e.g. "Map", "LaserScan"
    name: str
    enabled: bool
    topic: str | None
    props: dict = field(default_factory=dict)

    def get(self, key, default=None):
        value = self.props.get(key, default)
        return default if value is None else value

    def color(self, key="Color", default=(255, 255, 255)):
        return parse_color(self.props.get(key), default)

    def alpha(self, key="Alpha", default=1.0):
        try:
            return float(self.props.get(key, default))
        except (TypeError, ValueError):
            return default

    def nested(self, key, sub, default=None):
        value = self.props.get(key)
        if isinstance(value, dict):
            return value.get(sub, default)
        return default


@dataclass
class View:
    cls: str = "Orbit"
    yaw: float = 0.0
    pitch: float = 1.5707
    distance: float = 10.0
    focal_x: float = 0.0
    focal_y: float = 0.0


@dataclass
class RvizConfig:
    path: str
    fixed_frame: str = "map"
    background: tuple = (48, 48, 48)
    frame_rate: float = 30.0
    displays: list = field(default_factory=list)
    view: View = field(default_factory=View)

    @property
    def enabled_displays(self):
        return [d for d in self.displays if d.enabled]


def parse_color(value, default=(255, 255, 255)):
    """Parse RViz's ``"r; g; b"`` color string to an (r, g, b) tuple."""
    if not isinstance(value, str):
        return default
    try:
        parts = [int(float(p)) for p in value.split(";")]
    except ValueError:
        return default
    if len(parts) != 3:
        return default
    return tuple(max(0, min(255, p)) for p in parts)


def _topic_of(props):
    for key in ("Topic", "Description Topic"):
        value = props.get(key)
        if isinstance(value, dict):
            value = value.get("Value")
        if isinstance(value, str) and value:
            return value
    return None


def _flatten(displays):
    for raw in displays or []:
        if not isinstance(raw, dict):
            continue
        cls = str(raw.get("Class", ""))
        if cls.endswith("/Group"):
            if raw.get("Enabled", True):
                yield from _flatten(raw.get("Displays"))
            continue
        yield Display(
            cls=cls.split("/")[-1],
            name=str(raw.get("Name", cls)),
            enabled=bool(raw.get("Enabled", True)),
            topic=_topic_of(raw),
            props=raw,
        )


def load(path):
    """Load an .rviz file."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    vm = doc.get("Visualization Manager", {}) or {}
    opts = vm.get("Global Options", {}) or {}
    cfg = RvizConfig(path=path)
    cfg.fixed_frame = str(opts.get("Fixed Frame", "map")).lstrip("/")
    cfg.background = parse_color(opts.get("Background Color"), (48, 48, 48))
    cfg.frame_rate = float(opts.get("Frame Rate", 30) or 30)
    cfg.displays = list(_flatten(vm.get("Displays")))
    current = ((vm.get("Views") or {}).get("Current")) or {}
    focal = current.get("Focal Point") or {}
    cfg.view = View(
        cls=str(current.get("Class", "Orbit")).split("/")[-1],
        yaw=float(current.get("Yaw", 0.0) or 0.0),
        pitch=float(current.get("Pitch", 1.5707) or 0.0),
        distance=float(current.get("Distance", current.get("Scale", 10.0)) or 10.0),
        focal_x=float(focal.get("X", 0.0) or 0.0),
        focal_y=float(focal.get("Y", 0.0) or 0.0),
    )
    return cfg


def default_config_path():
    """Return the workspace's model.rviz (installed share first, then the source tree)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("mecanumbot_description")
        candidate = os.path.join(share, "rviz", "model.rviz")
        if os.path.isfile(candidate):
            return candidate
    except Exception:  # noqa: BLE001 - ament index missing or package not installed
        pass
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None
