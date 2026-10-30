#!/usr/bin/python3
"""Republish the LED state as a topic, so that a rosbag can record it.

ROS 2 Humble cannot record services, and the LED node only offers services, so
this calls get_led_status periodically and publishes each answer. The time of a
state is the message's time in the bag: there is no stamp in the message itself.

    /mecanumbot/led_status   std_msgs/msg/Int8MultiArray
    data = [fl_mode, fl_color, fr_mode, fr_color, br_mode, br_color, bl_mode, bl_color]

All -1 means the LED node could not read the Arduino's answer. What is reported is
the state last commanded to the Arduino, without the progress fill, which
get_led_status does not return.

Usage: led_status_poller.py [rate_hz]   (default 5)
"""
import sys

import rclpy
from mecanumbot_msgs.srv import GetLedStatus
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray

FIELDS = ("fl_mode", "fl_color", "fr_mode", "fr_color",
          "br_mode", "br_color", "bl_mode", "bl_color")
REQUEST_TIMEOUT_NS = 1_000_000_000


class LedStatusPoller(Node):
    def __init__(self, rate_hz):
        super().__init__("led_status_poller", namespace="/mecanumbot")
        self.client = self.create_client(GetLedStatus, "get_led_status")
        self.publisher = self.create_publisher(Int8MultiArray, "led_status", 10)
        self.pending = None
        self.pending_since = None
        self.was_available = None
        self.create_timer(1.0 / rate_hz, self.poll)

    def poll(self):
        available = self.client.service_is_ready()
        if available != self.was_available:
            if available:
                self.get_logger().info("get_led_status is up, polling")
            else:
                self.get_logger().warn("get_led_status is not available, nothing to record yet")
            self.was_available = available
        if not available:
            return

        now = self.get_clock().now().nanoseconds
        if self.pending is not None:
            if now - self.pending_since < REQUEST_TIMEOUT_NS:
                return
            self.client.remove_pending_request(self.pending)
        self.pending = self.client.call_async(GetLedStatus.Request())
        self.pending_since = now
        self.pending.add_done_callback(self.publish_answer)

    def publish_answer(self, future):
        self.pending = None
        if future.cancelled() or future.exception() is not None:
            return
        answer = future.result()
        self.publisher.publish(Int8MultiArray(data=[getattr(answer, f) for f in FIELDS]))


def main():
    rclpy.init()
    node = LedStatusPoller(float(sys.argv[1]) if len(sys.argv) > 1 else 5.0)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
