"""Color maps reproduced from rviz_default_plugins (map palettes, rainbow transformer)."""

import numpy as np

UNKNOWN_RGB = (0x70, 0x89, 0x86)


def bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _tail(palette):
    # 101..127: illegal positive values, drawn green; 128..254: illegal negative, red->yellow.
    palette[101:128] = (0, 255, 0, 255)
    for i in range(128, 255):
        palette[i] = (255, (255 * (i - 128)) // (254 - 128), 0, 255)


def map_palette():
    """Return the (256, 4) RGBA palette of RViz's "map" color scheme, indexed by uint8(value)."""
    p = np.zeros((256, 4), dtype=np.uint8)
    for i in range(101):
        v = 255 - (255 * i) // 100
        p[i] = (v, v, v, 255)
    _tail(p)
    p[255] = (*UNKNOWN_RGB, 255)
    return p


def costmap_palette():
    """Return the (256, 4) RGBA palette of RViz's "costmap" color scheme."""
    p = np.zeros((256, 4), dtype=np.uint8)
    p[0] = (0, 0, 0, 0)
    for i in range(1, 99):
        v = (255 * i) // 100
        p[i] = (v, 0, 255 - v, 255)
    p[99] = (0, 255, 255, 255)   # inscribed obstacle
    p[100] = (255, 0, 255, 255)  # lethal obstacle
    _tail(p)
    p[255] = (*UNKNOWN_RGB, 0)
    return p


def raw_palette():
    p = np.zeros((256, 4), dtype=np.uint8)
    p[:, 0] = p[:, 1] = p[:, 2] = np.arange(256)
    p[:, 3] = 255
    return p


PALETTES = {"map": map_palette, "costmap": costmap_palette, "raw": raw_palette}


def palette(scheme):
    return PALETTES.get(str(scheme).lower(), map_palette)()


def rainbow(values):
    """Vectorised rviz getRainbowColor: values in [0, 1] -> (N, 3) RGB floats in [0, 1]."""
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    h = v * 5.0 + 1.0
    i = np.floor(h).astype(int)
    f = h - i
    f = np.where(i % 2 == 0, 1.0 - f, f)
    n = 1.0 - f
    out = np.zeros((v.size, 3))
    one, zero = np.ones_like(n), np.zeros_like(n)
    cases = [
        (i <= 1, np.stack([n, zero, one], 1)),
        (i == 2, np.stack([zero, n, one], 1)),
        (i == 3, np.stack([zero, one, n], 1)),
        (i == 4, np.stack([n, one, zero], 1)),
        (i >= 5, np.stack([one, n, zero], 1)),
    ]
    for mask, col in cases:
        out[mask] = col[mask]
    return out


def scalar_colors(values, display, lo=None, hi=None):
    """
    Color a scalar channel the way RViz's Intensity/AxisColor transformers do.

    Return (N, 3) uint8 BGR.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if lo is None or hi is None:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    diff = hi - lo if hi > lo else 1.0
    norm = np.clip((values - lo) / diff, 0, 1)
    if display.get("Use rainbow", True):
        val = 1.0 - norm
        if display.get("Invert Rainbow", False):
            val = 1.0 - val
        rgb = rainbow(val)
    else:
        cmin = np.asarray(display.color("Min Color", (0, 0, 0)), float) / 255
        cmax = np.asarray(display.color("Max Color", (255, 255, 255)), float) / 255
        rgb = cmin + norm[:, None] * (cmax - cmin)
    return (rgb[:, ::-1] * 255).astype(np.uint8)
