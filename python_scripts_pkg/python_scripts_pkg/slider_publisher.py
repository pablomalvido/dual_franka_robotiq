import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import tkinter as tk
from geometry_msgs.msg import WrenchStamped


class MultiSliderPublisher(Node):
    def __init__(self):
        super().__init__('multi_slider_publisher')

        # Create publishers for each topic
        self.publisher_a = self.create_publisher(WrenchStamped, '/cartesian_force_controller/target_wrench', 10)
        self.publisher_b = self.create_publisher(WrenchStamped, '/cartesian_force_controller/ft_sensor_wrench', 10)

        # Create the GUI
        self.root = tk.Tk()
        self.root.title("ROS 2 Multi-Slider Publisher")

        # Slider A
        self.label_a = tk.Label(self.root, text="target force Z")
        self.label_a.pack()
        self.slider_a = tk.Scale(self.root, from_=-80, to=80,
                                 orient=tk.HORIZONTAL, command=self.publish_target)
        self.slider_a.pack()

        # Slider B
        self.label_b = tk.Label(self.root, text="Sensor force Z")
        self.label_b.pack()
        self.slider_b = tk.Scale(self.root, from_=-80, to=80,
                                 orient=tk.HORIZONTAL, command=self.publish_sensor)
        self.slider_b.pack()

    def publish_target(self, value):
        msg = WrenchStamped()
        msg.wrench.force.z = float(value)/10
        self.publisher_a.publish(msg)
        self.get_logger().info(f'[A] Published to /target_wrench: {msg.data}')

    def publish_sensor(self, value):
        msg = WrenchStamped()
        msg.wrench.force.z = float(value)/10
        self.publisher_b.publish(msg)
        self.get_logger().info(f'[B] Published to /tf_sensor_wrench: {msg.data}')

    def run(self):
        self.root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = MultiSliderPublisher()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()