#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import struct
import socket
from geometry_msgs.msg import WrenchStamped, Vector3
from std_msgs.msg import *
from std_srvs.srv import Trigger
import time
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R
import numpy as np

class NordboSensor(Node):
    def __init__(self):
        super().__init__('nordbo_ft_sensor_node')

        # Declare and get parameters
        self.declare_parameter('ip_address', '192.168.1.43')
        self.declare_parameter('port', 2001)
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('namespace_prefix', '')
        self.declare_parameter('target_link', '')
        self.declare_parameter('sensor_link', 'nordbo_sensor_frame')

        self.ip_address = self.get_parameter('ip_address').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.prefix = self.get_parameter('namespace_prefix').get_parameter_value().string_value
        self.target_link = self.get_parameter('target_link').get_parameter_value().string_value
        self.sensor_link = self.get_parameter('sensor_link').get_parameter_value().string_value

        # Create interfaces
        self.forces_pub = self.create_publisher(WrenchStamped, self.prefix+'nordbo/wrench', 10)
        self.forces_base_pub = self.create_publisher(WrenchStamped, self.prefix+'nordbo_robot_base/wrench', 10)
        self.tare_srv = self.create_service(Trigger, self.prefix+'nordbo/tare', self.tare_callback)
        self.abs_srv = self.create_service(Trigger, self.prefix+'nordbo/absolute', self.abs_callback)

        self.tare_signal = False
        self.abs_signal = False
        self.tare_forces = [0.0] * 6

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Constants
        self.CMD_TYPE_SENSOR_TRANSMIT = '07'
        self.SENSOR_TRANSMIT_TYPE_START = '01'
        self.SENSOR_TRANSMIT_TYPE_STOP = '00'
        self.CMD_TYPE_SET_CURRENT_TARE = '15'
        self.SET_CURRENT_TARE_TYPE_NEGATIVE = '01'
        self.SENSOR_SET_DATA_RATE = '08' #Specify the number of milliseconds between sent messages (1ms by default, 200ms max) 

        # Connect to sensor
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.settimeout(2.0)
            self.socket.connect((self.ip_address, self.port))
            self.get_logger().info(f"Connected to Nordbo force sensor at {self.ip_address}:{self.port}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to sensor: {e}")
            raise

        # Send tare command
        self.send_command('03' + self.CMD_TYPE_SET_CURRENT_TARE + self.SET_CURRENT_TARE_TYPE_NEGATIVE)
        self.recvMsg()

        # Start sensor stream
        self.send_command('03' + self.CMD_TYPE_SENSOR_TRANSMIT + self.SENSOR_TRANSMIT_TYPE_START)
        self.recvMsg()

        # Start sensor stream
        data_rate_hex = "{:02X}".format(int((1/self.publish_rate)*1000))
        self.send_command('03' + self.SENSOR_SET_DATA_RATE + data_rate_hex) #Freq, by default 1kHz
        self.recvMsg()

        # Start periodic reading
        self.create_timer(1.0 / (self.publish_rate), self.read_sensor_callback)

    def send_command(self, hex_str):
        try:
            self.socket.send(bytearray.fromhex(hex_str))
        except Exception as e:
            self.get_logger().error(f"Socket send error: {e}")

    def recvMsg(self):
        try:
            recv_data = bytearray(self.socket.recv(2))
            if len(recv_data) < 2:
                self.get_logger().warn(f"Received less than 2 bytes: {recv_data}")
                return bytearray()
            while len(recv_data) < recv_data[0]:
                recv_data += bytearray(self.socket.recv(recv_data[0] - len(recv_data)))
            return recv_data
        except Exception as e:
            self.get_logger().error(f"Socket recv error: {e}")
            return bytearray()

    def read_sensor_callback(self):
        recv_data = self.recvMsg()
        if len(recv_data) < 50:
            self.get_logger().warn("Incomplete sensor data received")
            self.get_logger().warn("Length of msg: " + str(len(recv_data)))
            return

        try:
            Fx = struct.unpack('!d', recv_data[2:10])[0]
            Fy = struct.unpack('!d', recv_data[10:18])[0]
            Fz = struct.unpack('!d', recv_data[18:26])[0]
            Tx = struct.unpack('!d', recv_data[26:34])[0]
            Ty = struct.unpack('!d', recv_data[34:42])[0]
            Tz = struct.unpack('!d', recv_data[42:50])[0]
            #self.get_logger().info("Sensor readings: Fx: " + str(Fx) + ", Fy: " + str(Fy) + ", Fz: " + str(Fz))
        except Exception as e:
            self.get_logger().error(f"Unpacking error: {e}")
            return

        if self.tare_signal:
            self.tare_forces = [Fx, Fy, Fz, Tx, Ty, Tz]
            self.tare_signal = False
            self.get_logger().info("Tared force sensor values")

        if self.abs_signal:
            self.tare_forces = [0.0] * 6
            self.abs_signal = False
            self.get_logger().info("Reset force sensor values to absolute")

        # Prepare message
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = self.sensor_link

        Fx_msg = Fx - self.tare_forces[0]
        Fy_msg = Fy - self.tare_forces[1]
        Fz_msg = Fz - self.tare_forces[2]
        Tx_msg = Tx - self.tare_forces[3]
        Ty_msg = Ty - self.tare_forces[4]
        Tz_msg = Tz - self.tare_forces[5]

        wrench_msg.wrench.force.x = Fx_msg
        wrench_msg.wrench.force.y = Fy_msg
        wrench_msg.wrench.force.z = Fz_msg
        wrench_msg.wrench.torque.x = Tx_msg
        wrench_msg.wrench.torque.y = Ty_msg
        wrench_msg.wrench.torque.z = Tz_msg

        self.forces_pub.publish(wrench_msg)

        #""" NOT NEEDED FOR NOW
        if(self.target_link != "" and self.sensor_link != ""):
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.target_link,
                    self.sensor_link,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.01)
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                return

            # Extract rotation as a 3x3 matrix
            q = trans.transform.rotation
            rot = R.from_quat([q.x, q.y, q.z, q.w])
            rot_matrix = rot.as_matrix()

            # Convert force and torque to numpy arrays
            force = np.array([Fx_msg, Fy_msg, Fz_msg])
            torque = np.array([Tx_msg, Ty_msg, Tz_msg])

            # Rotate force and torque vectors
            transformed_force = rot_matrix.dot(force)
            transformed_torque = rot_matrix.dot(torque)

            # Publish transformed wrench
            transformed_msg = WrenchStamped()
            transformed_msg.header.stamp = self.get_clock().now().to_msg()
            transformed_msg.header.frame_id = self.target_link

            transformed_msg.wrench.force.x = transformed_force[0]
            transformed_msg.wrench.force.y = transformed_force[1]
            transformed_msg.wrench.force.z = transformed_force[2]

            transformed_msg.wrench.torque.x = transformed_torque[0]
            transformed_msg.wrench.torque.y = transformed_torque[1]
            transformed_msg.wrench.torque.z = transformed_torque[2]

            self.forces_base_pub.publish(transformed_msg)
        #"""


    def tare_callback(self, request, response):
        self.tare_signal = True
        response.success = True
        response.message = "Sensor tared."
        return response

    def abs_callback(self, request, response):
        self.abs_signal = True
        response.success = True
        response.message = "Sensor set to absolute mode."
        return response

    def destroy_node(self):
        self.get_logger().info("Shutting down and stopping sensor stream...")
        try:
            self.send_command('03' + self.CMD_TYPE_SENSOR_TRANSMIT + self.SENSOR_TRANSMIT_TYPE_STOP)
            while True:
                recv = self.recvMsg()
                if len(recv) >= 2 and recv[0] == 3 and recv[1] == int(self.CMD_TYPE_SENSOR_TRANSMIT, 16):
                    break
        except Exception as e:
            self.get_logger().warn(f"Exception on shutdown: {e}")
        self.socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NordboSensor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()