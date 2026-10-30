# CLAUDE.md

Recorder and converter for MecanumBot ROS 2 bags. `bin/ros_record` records them; `bin/bag_convert` converts them: `render` (RViz config → top-down .mp4, offline),
`capture` (real RViz2 screen-recorded during `ros2 bag play`), `images` (image topics →
frames + index.csv and/or .mp4). The user-facing docs are in `README.md`.

## Run and test

- Always run with `/usr/bin/python3` (via `bin/bag_convert`). Conda's `python3` on PATH has no
  `rclpy`/`rosbag2_py`.
- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q` (`-p no:launch_testing`
  is no longer enough on this machine). The tests are pure Python: keep ROS imports deferred
  inside functions (`bag_io` does this) so they keep running without a sourced ROS.
- Lint: `ruff check .` (config in `pyproject.toml`).
- Use the scratchpad or `/tmp` for test outputs, never the bag directories (the leading study's
  are in `~/Videos/Ethology/Leading/bag_debug/`, whose `process_raw_footage.sh` calls `bin/bag_convert`).

## Layout

| Module | Role |
| --- | --- |
| `bag_io.py` | rosbag2_py reading, storage-id detection, deferred message imports |
| `tf_buffer.py` | whole-recording TF tree; interpolated lookups; robot↔recorder clock offset |
| `rviz_config.py` | `.rviz` YAML → `Display` list (groups flattened), fixed frame, view |
| `displays.py` | one class per RViz display type; `consume(msg)` / `draw(img, view)` |
| `render.py` | two-pass render loop, view fitting, HUD |
| `images.py` | image decoding (no cv_bridge, which is NumPy-1-only here), export, resampling |
| `capture.py` | Xvfb/Xephyr + rviz2 + `ros2 bag play` + ffmpeg x11grab |
| `video.py` | ffmpeg (libx264, yuv420p) writer |

`bin/ros_record` (the `ros_record` alias) records the usual topics into
`~/Videos/Ethology/Leading/bag_debug/`. Humble cannot record services, so it also runs
`bin/led_status_poller.py`, which republishes `get_led_status` as `/mecanumbot/led_status`
(`std_msgs/Int8MultiArray`, 5 Hz by default, `LED_POLL_HZ`). It imports ROS at the top, so keep it
out of the `bag_converter` package.

## Things that must stay true

- `capture` never records the host desktop unless the user passes `--display desktop`. x11grab
  records screen pixels, so an occluding window (e.g. a video call) ends up in the file. This
  happened once during development.
- `capture` never plays on `ROS_DOMAIN_ID=19` (the live robot's domain).
- `images --format original` writes compressed payloads byte-for-byte; do not re-encode them.
- Image videos are timed by header stamps when they are complete and monotonic. Receive times
  on the Wi-Fi-recorded camera topic bunch up and would stutter.
- The RobotModel display is a stand-in (box + TF children), not the URDF meshes. Say so
  wherever it matters.
