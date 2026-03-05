from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ft_sensor_node = Node(
        package="nordbo_ft_sensors",
        executable="run",
        output='screen',
         parameters=[{
                'ip_address': '192.168.1.43',
                'port': 2001,
                'publish_rate': 100.0,
                'namespace_prefix':'',
                'target_link': 'fr3_link0',
                'sensor_link': 'nordbo_ft_sensor_link', #'fr3_ft_sensor'
            }],
        remappings=[
                ('/nordbo_robot_base/wrench', '/cartesian_force_to_velocity_controller/current_force')
            ]
    )

    return LaunchDescription([ft_sensor_node])