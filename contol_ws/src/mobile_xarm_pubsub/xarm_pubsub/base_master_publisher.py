# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

import requests

from std_msgs.msg import Int32MultiArray
import time

try:
  from .my_vars import timer_period
except:
  from my_vars import timer_period

class BasePublisher(Node):

    def __init__(self):
        super().__init__('base_publisher')
        self.publisher_ = self.create_publisher(Int32MultiArray, '/base_vals', 10)
        self.declare_parameter('esp32_ip', value="http://192.168.1.211")
        self.esp32_ip = self.get_parameter('esp32_ip').value
        
        # timer_period = 0.75  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.sync_data=6*[500]
        self.prev_sync_data = 6*[500]
        self.start_time = time.time()

    def timer_callback(self):
        msg = Int32MultiArray()

        esp32_ip = self.esp32_ip
        resp=requests.get(esp32_ip+f"/get_encoders")
        
        # msg.header.stamp = self.get_clock().now().to_msg()
        if resp:
            resp = list(map(int,((resp.content).decode('utf-8')).split()))
            
            #return [int(raw_enc) for raw_enc in resp.split()]
        else:
            resp = [0, 0, 1, 1]
        
        msg.data = resp
        # self.get_logger().info("Wheel positions: %s" % ', '.join(str(pos) for pos in resp))
            
        self.publisher_.publish(msg)
        


def main(args=None):
    rclpy.init(args=args)

    base_publisher = BasePublisher()

    rclpy.spin(base_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    base_publisher.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
