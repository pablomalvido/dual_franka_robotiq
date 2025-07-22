from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ft_sensor_node = Node(
        package="nordbo_ft_sensors",
        executable="run",
        output='screen',
    )

    return LaunchDescription([ft_sensor_node])