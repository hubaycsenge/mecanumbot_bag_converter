"""
Offline TF buffer built from a bag's /tf and /tf_static.

Unlike tf2 it holds the whole recording, so any frame can be looked up at
any time. Dynamic edges are interpolated between the two samples around the
requested time and clamped (not extrapolated) outside the recorded range.
"""

import bisect

import numpy as np

from .geometry import invert, make_transform, slerp


class _Edge:
    def __init__(self, parent):
        self.parent = parent
        self.times = []
        self.trans = []
        self.quats = []
        self.static = None  # 4x4 when the edge came from /tf_static

    def finalize(self):
        if self.times:
            order = np.argsort(np.asarray(self.times), kind="stable")
            self.times = [self.times[i] for i in order]
            self.trans = np.asarray(self.trans, dtype=float)[order]
            self.quats = np.asarray(self.quats, dtype=float)[order]

    def at(self, t_ns):
        if self.static is not None:
            return self.static
        times = self.times
        i = bisect.bisect_left(times, t_ns)
        if i <= 0:
            return make_transform(self.trans[0], self.quats[0])
        if i >= len(times):
            return make_transform(self.trans[-1], self.quats[-1])
        t0, t1 = times[i - 1], times[i]
        f = 0.0 if t1 == t0 else (t_ns - t0) / (t1 - t0)
        tr = self.trans[i - 1] + f * (self.trans[i] - self.trans[i - 1])
        q = slerp(self.quats[i - 1], self.quats[i], f)
        return make_transform(tr, q)


def _norm(frame):
    return frame.lstrip("/")


class TfBuffer:
    """All transforms of a recording, keyed by child frame."""

    def __init__(self):
        self._edges = {}
        self._finalized = False
        # Median (receive time - header stamp) of /tf, used to map bag time
        # into the stamp domain when the robot clock differs from the recorder's.
        self.clock_offset_ns = 0
        self._offsets = []

    def add(self, transform, static=False, receive_ns=None):
        """Add one geometry_msgs/TransformStamped."""
        child = _norm(transform.child_frame_id)
        parent = _norm(transform.header.frame_id)
        tr = transform.transform.translation
        q = transform.transform.rotation
        edge = self._edges.get(child)
        if edge is None or edge.parent != parent:
            # tf allows one parent per child; the latest declaration wins.
            edge = _Edge(parent)
            self._edges[child] = edge
        if static:
            edge.static = make_transform((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))
        else:
            stamp = transform.header.stamp.sec * 1_000_000_000 + transform.header.stamp.nanosec
            edge.times.append(stamp)
            edge.trans.append((tr.x, tr.y, tr.z))
            edge.quats.append((q.x, q.y, q.z, q.w))
            if receive_ns is not None and stamp > 0:
                self._offsets.append(receive_ns - stamp)
        self._finalized = False

    def finalize(self):
        for edge in self._edges.values():
            edge.finalize()
        if self._offsets:
            self.clock_offset_ns = int(np.median(self._offsets))
        self._finalized = True

    @property
    def frames(self):
        out = set(self._edges)
        out.update(e.parent for e in self._edges.values())
        return out

    def children_of(self, frame):
        frame = _norm(frame)
        return [c for c, e in self._edges.items() if e.parent == frame]

    def has_dynamic(self, frame):
        edge = self._edges.get(_norm(frame))
        return edge is not None and edge.static is None

    def time_range(self, frame):
        """Return (first, last) stamp of a dynamic edge, or None."""
        edge = self._edges.get(_norm(frame))
        if edge is None or edge.static is not None or not edge.times:
            return None
        return edge.times[0], edge.times[-1]

    def _chain(self, frame, t_ns):
        """Return {ancestor: T_ancestor_frame} for the frame and all its ancestors."""
        out = {frame: np.eye(4)}
        acc = np.eye(4)
        seen = {frame}
        while frame in self._edges:
            edge = self._edges[frame]
            acc = edge.at(t_ns) @ acc
            frame = edge.parent
            if frame in seen:  # cycle guard
                break
            seen.add(frame)
            out[frame] = acc
        return out

    def lookup(self, target, source, t_ns):
        """Return T_target_source (maps source-frame points into target) or None."""
        if not self._finalized:
            self.finalize()
        target, source = _norm(target), _norm(source)
        if target == source:
            return np.eye(4)
        src_chain = self._chain(source, t_ns)
        if target in src_chain:
            return src_chain[target]
        tgt_chain = self._chain(target, t_ns)
        for frame, t_frame_source in src_chain.items():
            if frame in tgt_chain:
                return invert(tgt_chain[frame]) @ t_frame_source
        return None
