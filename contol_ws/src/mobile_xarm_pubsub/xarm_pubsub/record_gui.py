

import argparse
import random
import signal
import sys
import threading

import rclpy


import h5py
import os
import time

# import cv2
from rclpy.executors import MultiThreadedExecutor

# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMainWindow

from python_qt_binding.QtWidgets import QMainWindow
from python_qt_binding.QtWidgets import QPushButton
from python_qt_binding.QtWidgets import QVBoxLayout
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtWidgets import QApplication
from python_qt_binding.QtWidgets import QLabel
from python_qt_binding.QtGui import QPixmap, QImage
from python_qt_binding.QtCore import QTimer

from .puppet_subscriber import MinimalSubscriber #ps
from .camera_subscriber import CameraSubscriber #cs
from .base_camera_subscriber import BaseCameraSubscriber #bcs
from .base_puppet_subsriber import BaseSubscriber #ms
import logging
#need this or else opencv doesn't work with PyQt5
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'


class puppet_gui(QMainWindow):
    def __init__(self, title, ps,ms, cs, bcs):
        super(puppet_gui, self).__init__()
        self.setWindowTitle(title)


        # Button for randomizing the sliders
        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_episode)

        # Button for centering the sliders
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_episode)

        # Label to display the image
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)  # Set fixed size for the image label


        # Main layout
        self.main_layout = QVBoxLayout()

        # Add buttons and scroll area to main layout
        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.stop_button)
        self.main_layout.addWidget(self.image_label)

        # central widget
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # Update frame every 10 milliseconds

        # self.thread_gui = threading.Thread(target=self.root.mainloop(), daemon=False)
        self.ps=ps #puppet subscriber
        self.ms=ms #puppet subscriber
        self.cs = cs #camera subsriber
        self.bcs = bcs #base camera suscriber

        self.puppet_data = ps.data_dict
        self.image_data = cs.data_dict
        self.mobile = ms.data_dict
        self.base_image_data = bcs.data_dict


        
        self.strt_ep = False
        self.stp_ep = False

        self.record_params = ps.record_params
        self.record_paramscs = cs.record_params
        self.record_paramsms = ms.record_params
        self.record_paramsbcs = bcs.record_params

        self.running = True

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('puppet_gui')

        self.dataset_path = '~/projects/mobile_aloha_world/episodes/task1'#'~/tmp/my_episode'

    def stop_episode(self):
        self.record_params["stop"]=True
        self.record_params["start"] = False
        self.record_paramscs["stop"]=True
        self.record_paramscs["start"] = False
        self.record_paramsms["stop"]=True
        self.record_paramsms["start"] = False
        self.record_paramsbcs["stop"]=True
        self.record_paramsbcs["start"] = False
        time.sleep(0.1)
        self.record_episode()
        
    
    def start_episode(self):
        self.record_params["start"] = True
        self.record_paramscs["start"] = True
        self.record_paramsms["start"] = True
        self.record_paramsbcs["start"] = True
        

    def closeEvent(self, event):
        self.running = False

    def loop(self):
        while self.running:
            # rclpy.spin(self.ps)
            # self.ps.destory_node()
            # rclpy.spin(self.cs)
            rclpy.spin_once(self.ps, timeout_sec=0.1)
            

    def loopcs(self):
        while self.running:
            rclpy.spin_once(self.cs, timeout_sec=0.1)

    def loopbcs(self):
        while self.running:
            rclpy.spin_once(self.bcs, timeout_sec=0.1)

    def loopms(self):
        while self.running:
            rclpy.spin_once(self.ms, timeout_sec=0.1)
            
    def spin_nodes(self):
        executor = MultiThreadedExecutor()
        executor.add_node(self.ms)
        executor.add_node(self.ps)
        executor.add_node(self.cs)        
        executor.add_node(self.bcs)
        executor.spin()

    def update_frame(self):
        frame = self.cs.current_frame
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        # Convert QImage to QPixmap and display in QLabel
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        


    def get_auto_index(self, dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
        max_idx = 1000
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
        for i in range(max_idx+1):
            if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
                return i
        raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


    def record_episode(self):
        
        file_path = os.path.expanduser(self.dataset_path)
        i = self.get_auto_index(file_path)
        file_path = os.path.join(file_path, f'episode_{i}.hdf5')
        with h5py.File(file_path, 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            l1 = len(self.cs.data_dict['/observations/images/top'])
            l2 = len(self.bcs.data_dict['/observations/images/base'])
            la = len(self.ps.data_dict['/action'])
            lb = len(self.ms.data_dict['/base_action'])
            le = len(self.ms.data_dict['/observations/encoder_pos'])
            lq = len(self.ps.data_dict['/observations/qpos'])
            _ = image.create_dataset('top', (l1, 480, 640, 3), dtype='uint8', 
                                     chunks=(1,480,640,3), )
            _ = image.create_dataset('base', (l2, 240, 320, 3), dtype='uint8', 
                                     chunks=(1,240,320,3), )
            _ = obs.create_dataset('qpos', (lq, 6))
            _ = obs.create_dataset('encoder_pos', (le, 4))
            # _ = obs.create_dataset('qvel', (max_timesteps, 14))
            # _ = obs.create_dataset('effort', (max_timesteps, 14))
            _ = root.create_dataset('action', (la, 6))
            _ = root.create_dataset('base_action', (lb, 4))

            for name, array in self.ps.data_dict.items():
                self.logger.info(name)
                # print(array.shape())
                root[name][...] = array
            for name, array in self.cs.data_dict.items():
                self.logger.info(name)
                root[name][...] = array
            for name, array in self.ms.data_dict.items():
                root[name][...] = array
            for name, array in self.bcs.data_dict.items():
                root[name][...] = array

            #next clear the dictionary for next use.
            for key in self.ps.data_dict:
                self.ps.data_dict[key] = []
            for key in self.cs.data_dict:
                self.cs.data_dict[key] = []
            for key in self.ms.data_dict:
                self.ms.data_dict[key] = []
            for key in self.bcs.data_dict:
                self.bcs.data_dict[key] = []
        # print(f'Saving: {time.time() - t0:.1f} secs')

        return True


def main():
    # Initialize rclpy with the command-line arguments
    rclpy.init()

    # Strip off the ROS 2-specific command-line arguments
    stripped_args = rclpy.utilities.remove_ros_args(args=sys.argv)
    parser = argparse.ArgumentParser()
    # parser.add_argument('urdf_file', help='URDF file to use', nargs='?', default=None)

    # Parse the remaining arguments, noting that the passed-in args must *not*
    # contain the name of the program.
    parsed_args = parser.parse_args(args=stripped_args[1:])
    # print(parsed_args)

    app = QApplication(sys.argv)
    jsp_gui = puppet_gui('Episode Recorder',
                                    MinimalSubscriber(), 
                                    BaseSubscriber(),
                                    CameraSubscriber(),
                                    BaseCameraSubscriber()
                                    )

    jsp_gui.show()

    threading.Thread(target=jsp_gui.spin_nodes).start()
    # threading.Thread(target=jsp_gui.loopcs).start()
    # threading.Thread(target=jsp_gui.loop).start()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
