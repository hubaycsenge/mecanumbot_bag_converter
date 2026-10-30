# mecanumbot_bag_converter

Turn MecanumBot ROS 2 bags into media:

1. **RViz view → `.mp4`**: what `model.rviz` shows in `mecanumbot_ws` (map, laser scan,
   people detections, subject pose, paths, costmap, robot), rendered from the bag.
2. **Image streams → separate image output**: every `sensor_msgs/Image` /
   `CompressedImage` topic exported as its own folder of frames (+ `index.csv`), and
   optionally its own `.mp4`.

```bash
bin/bag_convert info    ~/rosbag2_2026_10_16-09_47_02   # topics + which RViz displays will be drawn
bin/bag_convert render  ~/rosbag2_2026_10_16-09_47_02   # -> rosbag2_..._rviz.mp4
bin/bag_convert images  ~/rosbag2_2026_10_16-09_47_02 --video   # -> rosbag2_..._images/
bin/bag_convert capture ~/rosbag2_2026_10_16-09_47_02   # real RViz, screen-recorded
```

`bin/bag_convert` runs the package with `/usr/bin/python3` (ROS 2 Humble's), because conda's
`python3` cannot import `rclpy`/`rosbag2_py`. It needs a shell with ROS 2 and
`mecanumbot_ws` sourced, which `~/.bashrc` already does. Sourcing the workspace is what makes
`mecanumbot_msgs` types importable; topics whose type cannot be imported are skipped with a
warning. Bags may be given as the directory, its `metadata.yaml` or a `.db3`/`.mcap` file.

## Recording

```bash
bin/ros_record                  # the `ros_record` alias
LED_POLL_HZ=10 bin/ros_record   # read the LED state more often
```

Records the usual topics into `~/Videos/Ethology/Leading/bag_debug/`; further options go to
`ros2 bag record`. Humble cannot record services, so `bin/led_status_poller.py` calls
`get_led_status` (5 Hz by default) and publishes each answer on `/mecanumbot/led_status`
(`std_msgs/Int8MultiArray`), whose time in the bag is the state's time:

    [fl_mode, fl_color, fr_mode, fr_color, br_mode, br_color, bl_mode, bl_color]

All `-1` means the LED node could not read the Arduino's answer. The values are the state last
commanded to the Arduino, without the progress fill, which `get_led_status` does not return.

## 1. RViz → mp4

There are two ways to do it.

### `render` (default choice): offline, headless, frame-exact

Reads the `.rviz` config (default: `mecanumbot_description/rviz/model.rviz`, from the
installed share, else the source tree) and redraws every enabled display top-down in the
config's fixed frame, using the display's own topic, colours, alpha, sizes, decay time and
colour scheme. It loads the whole of `/tf` + `/tf_static` first, so each message is placed with
the transform at its own stamp (interpolated). Then it streams the bag and writes one frame per
tick of bag time. It runs faster than real time (≈1 s per second of bag at 1080p30) and needs
no display.

| Display class | Drawn as |
| --- | --- |
| Map | occupancy grid with RViz's `map` / `costmap` / `raw` palettes, map updates applied |
| LaserScan, PointCloud2 | points; Intensity / RGB8 / AxisColor / FlatColor transformers; Decay Time |
| Pose, PoseArray, Odometry | arrows (or axes) with the configured lengths and colours |
| Path | polyline |
| Grid | grid lines in its reference frame |
| RobotModel | **stand-in**: body box, wheels and TF children of `base_link` (no URDF meshes) |
| TF | axes for every frame |

Other classes (Marker, Image, …) are listed as "not drawn". `info` prints, for each display
in the config, whether it will be drawn, is disabled, or has no topic in the bag.

```bash
bin/bag_convert render BAG -o out.mp4 \
    --view fit|follow|rviz   # whole map + trajectory | centred on robot | config's saved camera
    --view-size 6            # metres shown vertically in follow mode
    --rotation auto          # north-up for fit/follow, the config camera's yaw for --view rviz
    --start 120 --end 180    # seconds from bag start
    --speed 4 --fps 30 --size 1920x1080
    --remap /plan=/other_plan   # feed a display from another topic
    --no-hud --no-legend
```

The HUD shows the bag name, time since bag start and wall-clock time, plus a legend of the
drawn displays and a scale bar.

The renderer also estimates the robot clock's offset from the recorder's (median of receive
time − header stamp over `/tf`), and uses it to place stamp-less data and the robot at each
frame tick.

### `capture`: the real RViz2, screen-recorded

Starts `rviz2 -d model.rviz` with `use_sim_time`, publishes `/robot_description` from
`mecanumbot.urdf` (link names given the bag's `mecanumbot/` frame prefix), runs
`ros2 bag play --clock`, and records the RViz window with ffmpeg. This gives true RViz pixels:
meshes, every plugin, and the config's 3D camera. The trade-offs:

- it runs in real time (`--rate 0.5` plays slower and retimes the video back to real time);
- RViz renders in software on the private display (≈8 fps here), so motion is choppier than
  `render`;
- 1 cm laser points (the config's `Size (m)`) are invisible at the saved zoom, exactly as in
  RViz.

**Privacy.** RViz runs on a **private X server**: Xvfb if installed (invisible), else Xephyr
(a nested server that shows up as its own window). ffmpeg records that server's framebuffer, so
only RViz can end up in the video. `--display desktop` records the real screen instead.
x11grab copies screen pixels, so **anything covering the RViz window (a video call, a browser)
is recorded**. Use that mode only on a clear screen. `sudo apt install xvfb` makes capture
invisible.

The replay runs on `ROS_DOMAIN_ID=77` (`--domain-id`; 19, the robot's domain, is refused), so
it never reaches a live robot. With `--start`, latched topics (`/map`, `/tf_static`) are
replayed first, because `--start-offset` would otherwise skip the single `/map` message at
the bag's start.

## 2. Image streams

```bash
bin/bag_convert images BAG [-t TOPIC ...] [-o DIR] [--format original|png|jpg]
                           [--video] [--no-frames] [--fps N] [--every N]
                           [--start S] [--end S] [--header-stamp]
```

Each topic gets `DIR/<topic with / → __>/`:

- `frame_000000_<bag_time_ns>.jpg` …: with `--format original` (default), compressed payloads
  are written **byte-for-byte**, with no re-encoding. Raw `Image` messages are decoded (rgb8,
  bgr8, rgba/bgra, mono8/16, 16UC1, 32FC1, yuv422, bayer) and written as PNG; float data
  goes to TIFF. `compressedDepth` PNGs are unwrapped.
- `index.csv`: `index, file, bag_time_ns, header_stamp_ns, frame_id, width, height, encoding`.
  Width/height are blank for compressed images that were not decoded.
- `--video` adds `DIR/<topic>.mp4`, and `--no-frames` writes only the video. The video is a
  real-time timeline: it is timed by **header stamps** (the camera's capture times) when every
  frame has one and they are monotonic, otherwise by bag receive time. Frames are repeated
  across gaps, at 30 fps or the topic's mean rate if higher. The camera topics recorded over
  Wi-Fi arrive in bursts (receive gaps of 0 ms … 1.5 s against stamp gaps of ≥50 ms), so
  receive-time timing would stutter and drop frames. The summary reports any frames that still
  fell between ticks.

cv_bridge is deliberately not used: Humble's build on this machine is compiled against
NumPy 1.x and fails to import under the installed NumPy 2.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q   # 23 tests, no ROS graph, no bags
ruff check .
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed because the sourced ROS 2 environment injects
`launch_testing` pytest plugins that break collection.
