"""Small rigid-transform helpers (4x4 homogeneous matrices, xyzw quaternions)."""

import math

import numpy as np


def quat_to_mat(x, y, z, w):
    """Return the 3x3 rotation matrix of an xyzw quaternion (normalised first)."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_transform(translation, quaternion):
    """Return a 4x4 matrix from a (x, y, z) translation and an (x, y, z, w) quaternion."""
    t = np.eye(4)
    t[:3, :3] = quat_to_mat(*quaternion)
    t[:3, 3] = translation
    return t


def pose_to_mat(pose):
    """Return the 4x4 matrix of a geometry_msgs/Pose."""
    p, q = pose.position, pose.orientation
    return make_transform((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def invert(t):
    """Invert a rigid 4x4 transform."""
    inv = np.eye(4)
    r = t[:3, :3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t[:3, 3]
    return inv


def slerp(q0, q1, f):
    """Spherically interpolate between two xyzw quaternions."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    s = math.sin(theta)
    return (math.sin((1 - f) * theta) * q0 + math.sin(f * theta) * q1) / s


def transform_points(t, pts):
    """Apply a 4x4 transform to an (N, 3) array of points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ t[:3, :3].T + t[:3, 3]


def heading_xy(t):
    """Return the angle of the transform's x axis projected onto the XY plane."""
    return math.atan2(t[1, 0], t[0, 0])
