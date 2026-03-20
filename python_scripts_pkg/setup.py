from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'python_scripts_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        # Install config files
        (os.path.join('share', package_name, 'config'), glob('config/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='pmalvido7@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "interactive_marker_rviz = python_scripts_pkg.interactive_marker_rviz:main",
            "interactive_marker_rviz2 = python_scripts_pkg.interactive_marker_rviz2:main",
            "velocity_controller_gui = python_scripts_pkg.velocity_controller_gui:main",
            "force_controller_gui = python_scripts_pkg.force_controller_gui:main",
            "kaleel_app_test = python_scripts_pkg.kaleel_app_test:main",
            "kaleel_app_force_impulse = python_scripts_pkg.kaleel_app_force_impulse:main",
            "aruco_tracker = python_scripts_pkg.aruco_tracker_reference_d455_ros:main",
            "aruco_tracker_angle = python_scripts_pkg.aruco_tracker_reference_d455_ros_angle:main",
            "aruco_tracker_visibility = python_scripts_pkg.aruco_tracker_reference_d455_ros_visibility:main",
            "ses_pretension = python_scripts_pkg.ses_pretension:main",
            "ses_pretension_ring = python_scripts_pkg.ses_pretension_ring:main",
            "ses_pretension_angle = python_scripts_pkg.ses_pretension_angle:main",
            "ses_pretension_visibility = python_scripts_pkg.ses_pretension_visibility:main"
        ],
    },
)
