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



import threading
import queue
import time
import numpy as np
import cv2


from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import subprocess

from collections import deque

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = deque(maxlen=3)# queue.Queue()
    t = threading.Thread(target=self._reader, daemon=False)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      self.q.append(frame)
    #   if not self.q.empty():
    #     try:
    #       self.q.get_nowait()   # discard previous (unprocessed) frame
    #     except queue.Empty:
    #       pass
    #   self.q.put(frame)
      

  def read(self):
    return self.q.pop() #self.q.get()
#functions for the command handler



class BaseCameraPublisher(Node):

    def __init__(self):
        super().__init__('base_camera_publisher')
        URL_cam = "http://192.168.1.181"
        self.cap_esp32 = VideoCapture(URL_cam + ":81/stream")
        self.intel_publisher_rgb = self.create_publisher(Image, "base_rgb_frame", 10)
        timer_period = 0.5  # seconds
        self.br_rgb = CvBridge()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.start_time = time.time()

    def timer_callback(self):
        
        data = self.cap_esp32.read()
        cv2.imshow("RGB", data)
        cv2.waitKey(1)
        
        if data is not None:
            self.intel_publisher_rgb.publish(self.br_rgb.cv2_to_imgmsg(data))
            
        

        
        
        


def main(args=None):
    
    
    rclpy.init(args=args)

    base_camera_publisher = BaseCameraPublisher()

    rclpy.spin(base_camera_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    base_camera_publisher.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
