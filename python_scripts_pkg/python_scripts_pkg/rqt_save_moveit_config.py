#!/usr/bin/env python3

import sys
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
)

import xml.etree.ElementTree as ET


class JointStateListener(Node):
    def __init__(self):
        super().__init__('joint_state_listener')

        self.joint_positions = {}
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

    def joint_callback(self, msg):
        self.joint_positions = dict(zip(msg.name, msg.position))


class ConfigSaverGUI(QWidget):
    def __init__(self, ros_node, srdf_path):
        super().__init__()

        self.ros_node = ros_node
        self.srdf_path = srdf_path

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SRDF Config Saver")

        layout = QVBoxLayout()

        # Row: Label + Textbox
        row = QHBoxLayout()
        label = QLabel("config name")
        self.textbox = QLineEdit()

        row.addWidget(label)
        row.addWidget(self.textbox)

        layout.addLayout(row)

        # Button
        self.button = QPushButton("Save config")
        self.button.clicked.connect(self.save_config)

        layout.addWidget(self.button)

        self.setLayout(layout)

    def save_config(self):
        config_name = self.textbox.text().strip()

        if not config_name:
            print("Config name is empty")
            return

        joint_positions = self.ros_node.joint_positions

        if not joint_positions:
            print("No joint states received yet")
            return

        try:
            tree = ET.parse(self.srdf_path)
            root = tree.getroot()

            group_state = ET.Element('group_state')
            group_state.set('name', config_name)
            group_state.set('group', 'fr3_arm')  # adjust if needed

            for joint, value in joint_positions.items():
                joint_elem = ET.SubElement(group_state, 'joint')
                joint_elem.set('name', joint)
                joint_elem.set('value', str(value))

            root.append(group_state)

            ET.indent(tree, space="    ", level=0)

            tree.write(self.srdf_path, encoding='utf-8', xml_declaration=True)

            self.ros_node.get_logger().info(f"Saved config: {config_name}")

        except Exception as e:
            print(f"Error saving config: {e}")
            self.ros_node.get_logger().warn(f"Error saving config: {e}")


def main():
    rclpy.init()

    if len(sys.argv) < 2:
        print("Usage: script.py <path_to_srdf>")
        return

    srdf_path = sys.argv[1]

    print("SRDF path: " + srdf_path)

    ros_node = JointStateListener()
    ros_node.get_logger().info("SRDF path: " + srdf_path)

    # Run ROS in separate thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    gui = ConfigSaverGUI(ros_node, srdf_path)
    gui.show()

    exit_code = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()