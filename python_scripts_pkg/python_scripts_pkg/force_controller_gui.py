import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QSlider, QPushButton, QLabel
)
from PyQt5.QtCore import Qt

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3


class Vector3Publisher(Node):
    def __init__(self):
        super().__init__('force_vector3_gui_publisher')
        self.publisher_ = self.create_publisher(Vector3, '/cartesian_force_to_velocity_controller/target_force', 10)

    def publish_vector(self, x, y, z):
        msg = Vector3()
        msg.x = x
        msg.y = y
        msg.z = z
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published Vector3: x={x:.4f}, y={y:.4f}, z={z:.4f}')


class Vector3GUI(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node

        self.setWindowTitle('Force Vector3 Publisher GUI')
        self.setGeometry(100, 100, 300, 300)

        layout = QVBoxLayout()

        self.labels = {}
        self.sliders = {}
        self.values = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        for axis in ['x', 'y', 'z']:
            label = QLabel(f"{axis.upper()}: 0.0000")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-1000)
            slider.setMaximum(1000)
            slider.setValue(0)
            slider.valueChanged.connect(lambda val, ax=axis: self.update_value(ax, val))

            self.labels[axis] = label
            self.sliders[axis] = slider

            layout.addWidget(label)
            layout.addWidget(slider)

        self.publish_button = QPushButton('Publish')
        self.publish_button.clicked.connect(self.publish_vector)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_vector)

        layout.addWidget(self.publish_button)
        layout.addWidget(self.reset_button)

        self.setLayout(layout)

    def update_value(self, axis, value):
        float_val = value / 100.0  # Scale to -10.0 to 10.0
        self.values[axis] = float_val
        self.labels[axis].setText(f"{axis.upper()}: {float_val:.4f}")

    def publish_vector(self):
        self.ros_node.publish_vector(
            self.values['x'], self.values['y'], self.values['z']
        )

    def reset_vector(self):
        for axis in ['x', 'y', 'z']:
            self.sliders[axis].setValue(0)
        self.ros_node.publish_vector(0.0, 0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)

    node = Vector3Publisher()

    app = QApplication(sys.argv)
    gui = Vector3GUI(node)
    gui.show()

    # Spin ROS node in a background thread
    from threading import Thread
    ros_thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    app.exec_()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
