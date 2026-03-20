from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mujoco_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install config files
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rosdev',
    maintainer_email='pmalvido7@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ses_exp1 = mujoco_ros.ses_controller_real_exp1:main',
            'ses_exp2 = mujoco_ros.ses_controller_real_exp2:main',
            'ses_exp3 = mujoco_ros.ses_controller_real_exp3:main',
            'ses_real = mujoco_ros.ses_controller_real_generic:main',
            'ses_real_angle = mujoco_ros.ses_controller_real_generic_angle:main',
            'ses_real_visibility = mujoco_ros.ses_controller_real_generic_visibility:main'
        ],
    },
)
