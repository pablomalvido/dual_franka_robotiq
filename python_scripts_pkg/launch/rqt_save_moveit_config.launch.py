from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('python_scripts_pkg')

    srdf_file = os.path.join(
        pkg_share,
        'config',
        'test.srdf'
    )

    moveit_save_config_node = Node(
        package='python_scripts_pkg',
        executable='rqt_save_moveit_config',
        name='rqt_save_moveit_config',
        arguments=['/ros2_ws/src/developments/dual_franka_robotiq/python_scripts_pkg/config/fr3_ft_sensor.srdf.xacro'],#['-d', srdf_file],
        output='screen'
    )

    return LaunchDescription([
        moveit_save_config_node
    ])