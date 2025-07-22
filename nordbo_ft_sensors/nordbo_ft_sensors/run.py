#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import struct
import socket
from geometry_msgs.msg import WrenchStamped, Vector3
from std_msgs.msg import *
from std_srvs.srv import Trigger

class NordboSensor(Node):
    def __init__(self, prefix, ns):
        super().__init__('nordbo_ft_sensor_node')
        self.ns = ns
        self.prefix = prefix

        self.forces_pub = self.create_publisher(WrenchStamped, self.ns+'/'+self.prefix+'nordbo/wrench', 10)
        self.tare_signal = True
        self.abs_signal = False
        self.tare_forces = [0,0,0,0,0,0]
        self.tare_srv = self.create_service(Trigger, 'norbdo/tare', self.tare_callback)
        self.abs_srv = self.create_service(Trigger, 'norbdo/absolute', self.abs_callback)

        self.rate = self.create_rate(50)
        self.IP_ADDR = "192.168.1.43"
        self.PORT = 2001

        self.CMD_TYPE_SENSOR_TRANSMIT = '07'
        self.SENSOR_TRANSMIT_TYPE_START = '01'
        self.SENSOR_TRANSMIT_TYPE_STOP = '00'
        self.CMD_TYPE_SET_CURRENT_TARE = '15'
        self.SET_CURRENT_TARE_TYPE_NEGATIVE	= '01'

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)	
        self.init_sensor()
    

    def init_sensor(self):
        self.socket.settimeout(2.0)
        self.socket.connect((self.IP_ADDR, self.PORT))
        #print("Connection stablished with " + self.IP_ADDR + ":" + str(self.PORT))
        self.get_logger().info("Connection stablished with norbdo force sensor at " + self.IP_ADDR + ":" + str(self.PORT))

        sendData = '03' + self.CMD_TYPE_SET_CURRENT_TARE + self.SET_CURRENT_TARE_TYPE_NEGATIVE
        sendData = bytearray.fromhex(sendData)
        self.socket.send(sendData)
        recvData = self.recvMsg()

        sendData = '03' + self.CMD_TYPE_SENSOR_TRANSMIT + self.SENSOR_TRANSMIT_TYPE_START
        sendData = bytearray.fromhex(sendData)
        self.socket.send(sendData)
        recvData = self.recvMsg()

        i=0
        while rclpy.ok():
            recvData = self.recvMsg()
            Fx = struct.unpack('!d', recvData[2:10])[0]
            Fy = struct.unpack('!d', recvData[10:18])[0]
            Fz = struct.unpack('!d', recvData[18:26])[0]
            Tx = struct.unpack('!d', recvData[26:34])[0]
            Ty = struct.unpack('!d', recvData[34:42])[0]
            Tz = struct.unpack('!d', recvData[42:50])[0]
            if self.tare_signal:
                self.tare_forces = [float(Fx),float(Fy),float(Fz),float(Tx),float(Ty),float(Tz)]
                self.tare_signal = False
            if self.abs_signal:
                self.tare_forces = [0,0,0,0,0,0]
                self.abs_signal = False
            if i>=50: #Freq 1000/50 = 20Hz
                #print('Fx: '+str(Fx) + ", Fy: " + str(Fy) + ", Fz: " + str(Fz) + ", Tx: " + str(Tx) + ", Ty: " + str(Ty) + ", Tz: " + str(Tz))
                force_msg = Vector3()
                force_msg.x = float(Fx) - self.tare_forces[0]
                force_msg.y = float(Fy) - self.tare_forces[1]
                force_msg.z = float(Fz) - self.tare_forces[2]
                torque_msg = Vector3()
                torque_msg.x = float(Tx) - self.tare_forces[3]
                torque_msg.y = float(Ty) - self.tare_forces[4]
                torque_msg.z = float(Tz) - self.tare_forces[5]
                ft_msg = WrenchStamped()
                ft_msg.wrench.force = force_msg
                ft_msg.wrench.torque = torque_msg
                self.forces_pub.publish(ft_msg)
                i=0
            else:
                i+=1

        sendData = '03' + self.CMD_TYPE_SENSOR_TRANSMIT + self.SENSOR_TRANSMIT_TYPE_STOP
        sendData = bytearray.fromhex(sendData)
        self.socket.send(sendData)
        recvData = self.recvMsg()

        #Wait until and ACK msg is send back for the stop command. 
        while recvData[0] != 3 and recvData[1] != self.CMD_TYPE_SENSOR_TRANSMIT:
            recvData = self.recvMsg()

        self.socket.close()

    def recvMsg(self):
        recvData = bytearray(self.socket.recv(2))

        while len(recvData) < recvData[0]:
            recvData += bytearray(self.socket.recv(recvData[0] - len(recvData)))

        #self.printMsg(recvData)

        return recvData

    def printMsg(self, msg):
        print("Msg len: " + str(msg[0]) + " Msg type: " + str(msg[1]) + "")
        
        dataStr = "DATA: "
        for i in range(msg[0] - 2):
            dataStr += str(msg[i + 2]) + " "

        print(dataStr)


    def tare_callback(self, request, response):
        """
        Tares the sensor values
        """
        self.get_logger().info("Taring force sensor")
        self.tare_signal = True
        response.success = True
        return response
    

    def abs_callback(self, request, response):
        """
        Reset the sensor values
        """
        self.get_logger().info("Reseting force sensor values")
        self.abs_signal = True
        response.success = True
        return response 


def main(args=None):
    rclpy.init(args=args)
    node = NordboSensor('','')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()