import sys
import rclpy
import numpy as np
import time
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, Pose, Vector3, WrenchStamped, PoseArray, Point
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped
from custom_msgs.msg import SesStarter

class SESPretension(Node):

    def __init__(self):
        super().__init__('ses_pretension')

        self.log = []
        self.log_pos = []
        self.F_pretense = 2
        self.pretense = False
        self.ses_target = self.define_target_msg([
            (0.15, -0.23),
            (0.1, -0.1),
            (0.05, 0.03)])
        self.ee_length = 0.17

        self.write_data = False

        # Create TF buffer and listener to get the EE position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame = 'fr3_link0'
        self.ee_frame = 'fr3_link8'

        self.cli_switch_controller = self.create_client(SwitchController, 'controller_manager/switch_controller')
        self.cartesian_speed_controller_target_pub = self.create_publisher(Vector3, '/cartesian_velocity_controller_ses/target_speed', 10)
        self.ses_activate_pub = self.create_publisher(SesStarter, "ses/activate", 10)
        self.forces_sub = self.create_subscription(WrenchStamped, 'nordbo/wrench',self.read_forces,10) #1
        self.forces_sub
        
        #while not self.cli_target_config.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info('Move to target pose service not available, waiting...')
        while not self.cli_switch_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Switch controller service not available, waiting...')

        #self.move_target_req = MoveTarget.Request()
        self.switch_controller_req = SwitchController.Request()

    def move_target(self, type, pose=Pose(), config=""):
        if type == 'P':
            self.move_target_req.pose = pose
            return self.cli_target_pose.call_async(self.move_target_req)
        elif type == 'C' and not (config==""):
            self.move_target_req.config = config
            return self.cli_target_config.call_async(self.move_target_req)
        else:
            print("ERROR")
            return
        
    def switch_controller(self, start_c, stop_c):
        self.switch_controller_req.start_controllers = start_c
        self.switch_controller_req.stop_controllers = stop_c
        self.switch_controller_req.strictness = SwitchController.Request.STRICT
        self.switch_controller_req.timeout = Duration(sec=2, nanosec=0)
        return self.cli_switch_controller.call_async(self.switch_controller_req)
    
    def set_target_cartesian_speed(self, vx, vy, vz, t, timer=False):
        msg = Vector3()
        msg.x = vx
        msg.y = vy
        msg.z = vz
        self.cartesian_speed_controller_target_pub.publish(msg)
        if timer:
            self.start_time=time.time()
            self.timer=self.create_timer(t, self.timer_callback)
            self.write_data = True

    def timer_callback(self):
        # Stop the timer so it runs only once
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        self.set_target_cartesian_speed(0.0,0.0,0.0,0.0,False)  
        
    def read_forces(self,msg):
        Force=np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        F_total = np.linalg.norm(Force)
        print(F_total)
        if self.write_data:
            self.log.append({'fx': msg.wrench.force.x, 'fy': msg.wrench.force.y, 'fz': msg.wrench.force.z, 'F': F_total, 't': (time.time()-self.start_time)})
        if (F_total >= self.F_pretense) and not self.pretense:
            self.pretense = True
            self.set_target_cartesian_speed(0.0,0.0,0.0,0.0,False)
            self.get_logger().info("Pre-tension force achieved, changing to SES controller")
            self.timer_ses = self.create_timer(0.1, self.start_ses) 

    def get_tf_ee(self):
        try:
            # Lookup transform from base -> end effector
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time()
            )
            translation = transform.transform.translation
            #rotation = transform.transform.rotation
            return translation, True
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Transform not available: {e}")
            return [], False
        

    def start_ses(self):
        # Stop the timer so it runs only once
        if self.timer_ses is not None:
            self.timer_ses.cancel()
            self.timer_ses = None
        ee_pos, success = self.get_tf_ee()
        if success:
            msg = SesStarter()
            msg.start_position = Point()
            msg.start_position.x = ee_pos.x
            msg.start_position.y = ee_pos.y
            msg.start_position.z = ee_pos.z - self.ee_length
            msg.ses_target = self.ses_target
            self.ses_activate_pub.publish(msg)
            self.get_logger().info("SES started with EE position: x={:.3f}, y={:.3f}, z={:.3f}".format(ee_pos.x, ee_pos.y, (ee_pos.z-self.ee_length)))
        else:
            self.get_logger().warn(f"SES cannot be started because EE position is not available")

    def define_target_msg(self, target_list):
        # Create default PoseArray
        target_poses_msg = PoseArray()
        target_poses_msg.header.frame_id = "camera_link"
        target_poses_msg.header.stamp = self.get_clock().now().to_msg()

        for pos in target_list:
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = 0.0

            # Default orientation (identity quaternion)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0

            target_poses_msg.poses.append(pose)
        
        return target_poses_msg


def main(args=None):
    rclpy.init(args=args)

    node = SESPretension()

    stop_controllers = ['fr3_arm_controller']
    start_controllers = ['cartesian_velocity_controller_ses']

    node.get_logger().info("Switching controllers...")
    future3 = node.switch_controller(start_c=start_controllers, stop_c=stop_controllers)
    rclpy.spin_until_future_complete(node, future3)
    if future3.result() is not None:
        node.get_logger().info("Success")
    else:
        node.get_logger().error('Service call failed')

    node.get_logger().info("Setting cartesian controller target...")
    node.set_target_cartesian_speed(0.02,0.0,0.0,0.0,False)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()