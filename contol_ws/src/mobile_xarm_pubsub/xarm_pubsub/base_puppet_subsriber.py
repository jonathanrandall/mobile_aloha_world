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




class BaseSubscriber(Node):

    def __init__(self):
        super().__init__('base_subscriber')
        self.declare_parameter('esp32_ip', value="http://192.168.1.182")
        self.esp32_ip = self.get_parameter('esp32_ip').value

        self.declare_parameter('topic_name', '/base_vals')
        topic_name = self.get_parameter('topic_name').value
        self.subscription = self.create_subscription(
            Int32MultiArray,
            topic_name, 
            self.listener_callback, 
            10)
        self.subscription  # prevent unused variable warning
        self.start_time = time.time()
        

        self.data_dict = {
            '/observations/encoder_pos': [],
            # '/observations/qvel': [],
            # '/observations/effort': [],
            '/base_action': [],
            }
        
        self.encoder_pos = [0, 0, 0]
        
        self.record_params = {"stop": False, "start":False}

        self.max_timesteps=0
        


    

    def listener_callback(self, msg):
        # self.mutex.acquire()
        
        
        positions = msg.data
        # Convert the list of positions to a string
        positions_str = '_'.join(str(pos) for pos in positions)
        positions_int = [int(x) for x in positions]

        self.encoder_pos = positions_int

        esp32_ip = self.esp32_ip
        try:
            resp=requests.get(esp32_ip+f"/set_encoders?var=variable&val="+positions_str,timeout=2)         
        except requests.exceptions.Timeout:
            self.get_logger().info("Timed out")
            resp.status_code = 0
            resp.content="error"
            # exit()

        # print(resp.status_code)
        self.get_logger().info(resp.content)
        # self.get_logger().info("Base subscriber positions: %s" %  positions_str)#', '.join(str(pos) for pos in positions_int))
        
        
        if self.record_params["start"]:
            self.max_timesteps=self.max_timesteps+1
            self.data_dict['/observations/encoder_pos'].append(positions_int[:4])
            self.get_logger().info("Joint positions: %s" % ', '.join(str(pos) for pos in positions_int))
            self.data_dict['/base_action'].append(positions_int[:4])
        if self.record_params["stop"]: #(t2-self.start_time > 20):
            # stp_ep = False
            # strt_ep = False
            self.record_params["stop"]=False
            self.record_params["start"]=False
            self.get_logger().info(f"episode recoreded {len(self.data_dict['/base_action'])}")
            
            time.sleep(1)
            self.max_timesteps=1
        # self.mutex.release()


    

def main(args=None):
    # gui = myGUI()
    
    # gui.thread_gui.start()
    rclpy.init(args=args)

    base_subscriber = BaseSubscriber()
    

    rclpy.spin(base_subscriber)
    # gui.root.mainloop()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    base_subscriber.destroy_node()
    rclpy.shutdown()

    # thread_node.join()

    
    
    # thread_gui = threading.Thread(target=gui.root.mainloop, daemon=False).start()
    


if __name__ == '__main__':
    main()
