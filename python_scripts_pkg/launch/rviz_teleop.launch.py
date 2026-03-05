from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('python_scripts_pkg')

    rviz_config = os.path.join(
        pkg_share,
        'config',
        'franka_teleop.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    interactive_marker_node = Node(
        package='python_scripts_pkg',
        executable='interactive_marker_rviz',
        name='interactive_marker_rviz',
        output='screen'
    )

    return LaunchDescription([
        rviz_node,
        interactive_marker_node
    ])