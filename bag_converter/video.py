"""Write BGR frames to an H.264 .mp4 through ffmpeg (OpenCV mp4v as a fallback)."""

import os
import shutil
import subprocess
import sys


class VideoWriter:
    def __init__(self, path, width, height, fps, crf=20, preset="medium"):
        self.path = path
        self.width = int(width) - int(width) % 2   # yuv420p needs even sizes
        self.height = int(height) - int(height) % 2
        self.fps = float(fps)
        self.frames = 0
        self._proc = None
        self._cv = None
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:g}", "-i", "-",
                "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            import cv2
            print("warning: ffmpeg not found, writing mp4v with OpenCV", file=sys.stderr)
            self._cv = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height))

    def write(self, frame):
        import numpy as np

        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._proc is not None:
            self._proc.stdin.write(frame.tobytes())
        else:
            self._cv.write(frame)
        self.frames += 1

    def close(self):
        if self._proc is not None:
            self._proc.stdin.close()
            if self._proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
            self._proc = None
        elif self._cv is not None:
            self._cv.release()
            self._cv = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
