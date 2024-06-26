from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'xarm_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jonny',
    maintainer_email='j@u',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'master = xarm_pubsub.master_publisher:main',
            'puppet = xarm_pubsub.puppet_subscriber:main',
            'base_master = xarm_pubsub.base_master_publisher:main',
            'base_puppet = xarm_pubsub.base_puppet_subscriber:main',
            'record_gui = xarm_pubsub.record_gui:main',
            'image_pub = xarm_pubsub.camera_publisher:main',
            'image_sub = xarm_pubsub.camera_subscriber:main',
            'base_image_pub = xarm_pubsub.base_camera_publisher:main',
            'base_image_sub = xarm_pubsub.base_camera_subscriber:main',
        ],
    },
)
