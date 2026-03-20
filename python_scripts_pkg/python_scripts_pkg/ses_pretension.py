import sys
import rclpy
import numpy as np
import time
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, Pose, Vector3, WrenchStamped, PoseArray, Point
from std_msgs.msg import Header, Bool
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped
from custom_msgs.msg import SesStarter

class SESPretension(Node):

    def __init__(self, use_pretense=True):
        super().__init__('ses_pretension')

        self.log = []
        self.log_pos = []
        self.pretense = False
        self.use_pretense = use_pretense
        
        ## EXP1: TPU beam
        # self.F_pretense = 2
        # self.ses_target = self.define_target_msg([
        #     (0.15, -0.23),
        #     (0.1, -0.1),
        #     (0.05, 0.03)])
        # self.ses_target = self.define_target_msg([
        #     (0.2, -0.16),
        #     (0.05, -0.1),
        #     (0.01, 0.03)])
        
        ## EXP2: Chain
        self.F_pretense = 1.5
        # self.ses_target = self.define_target_msg([
        #     (-0.08, -0.23),
        #     (0.01, -0.16),
        #     (0.09, 0.01)])
        # self.ses_target = self.define_target_msg([
        #     (0.0, -0.26),
        #     (0.0, -0.12),
        #     (0.0, 0.03)])
        # self.ses_target = self.define_target_msg([
        #     (-0.04, -0.23),
        #     (0.02, -0.09),
        #     (0.02, 0.07)])
        ## Forward straight
        # self.ses_target = self.define_target_msg([
        #     (-0.04, -0.1),
        #     (-0.04, 0.04),
        #     (-0.04, 0.2)])
        ## Middle curvature
        # self.ses_target = self.define_target_msg([
        #     (-0.05, -0.12),
        #     (0.03, -0.02),
        #     (0.07, 0.17)])
        ## Bottom curvature: DIDNT WORK
        # self.ses_target = self.define_target_msg([
        #     (-0.16, -0.09),
        #     (-0.09, -0.04),
        #     (0.08, 0.0)])
        ## Bottom straight
        # self.ses_target = self.define_target_msg([
        #     (-0.16, -0.07),
        #     (-0.12, 0.06),
        #     (-0.07, 0.22)])
        ## Impossible
        # self.ses_target = self.define_target_msg([
        #     (-0.08, -0.10),
        #     (0.02, 0.05),
        #     (-0.08, 0.2)])

        ## EXP4: Branch
        ## Top bending 1,2,3
        # self.ses_target = self.define_target_msg([
        #     (0.11, -0.15),
        #     (-0.06, 0.03),
        #     (-0.13, 0.26)])
        ## Bottom bending
        # self.ses_target = self.define_target_msg([ 
        #     (0.17, -0.01),
        #     (-0.06, 0.12),
        #     (-0.23, 0.29)])
        ## Top bending
        # self.ses_target = self.define_target_msg([
        #     (0.02, -0.13),
        #     (-0.17, 0.05),
        #     (-0.28, 0.26)])
        ## Top bending branch 2
        self.ses_target = self.define_target_msg([
            (0.02, -0.001),
            (-0.14, 0.09),
            (-0.24, 0.22)])
        
        self.ee_length = 0.0

        self.write_data = False

        # Create TF buffer and listener to get the EE position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame = 'fr3_link0'
        self.ee_frame = 'nordbo_ft_sensor_link' #'fr3_link8'

        self.cli_switch_controller = self.create_client(SwitchController, 'controller_manager/switch_controller')
        self.cartesian_speed_controller_target_pub = self.create_publisher(Vector3, '/cartesian_velocity_controller_ses/target_speed', 10)
        self.ses_activate_pub = self.create_publisher(SesStarter, "ses/activate", 10)
        self.forces_sub = self.create_subscription(WrenchStamped, 'nordbo/wrench',self.read_forces,10) #1
        self.forces_sub
        self.SES_done = self.create_subscription(Bool, 'ses/finish',self.ses_done_cb,10) #1
        self.SES_done
        
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
        self.get_logger().info("Switching controllers...")
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
        if self.use_pretense:
            if (F_total >= self.F_pretense) and not self.pretense:
                self.pretense = True
                self.set_target_cartesian_speed(0.0,0.0,0.0,0.0,False)
                self.get_logger().info("Pre-tension force achieved, changing to SES controller")
                self.timer_ses = self.create_timer(0.1, self.start_ses) 

    def ses_done_cb(self, msg):
        self.get_logger().info("SES finished, stopping robot")
        start_controllers = []
        stop_controllers = ['cartesian_velocity_controller_ses']
        future4 = self.switch_controller(start_c=start_controllers, stop_c=stop_controllers)
        future4.add_done_callback(self.switch_done_1_cb)

    
    def switch_done_1_cb(self, future):
        try:
            if future.result() is not None:
                self.get_logger().info("Success")
            else:
                self.get_logger().error('Service call failed')
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        #Active MoveIt controller
        stop_controllers = []
        start_controllers = ['fr3_arm_controller']
        self.switch_controller(start_c=start_controllers, stop_c=stop_controllers)


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
        
    def direct_start_ses(self):
        self.get_logger().info("Changing to SES controller")
        self.timer_ses = self.create_timer(0.1, self.start_ses) 

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

    use_pretense = False
    node = SESPretension(use_pretense)

    stop_controllers = ['fr3_arm_controller']
    start_controllers = ['cartesian_velocity_controller_ses']

    future3 = node.switch_controller(start_c=start_controllers, stop_c=stop_controllers)
    rclpy.spin_until_future_complete(node, future3)
    if future3.result() is not None:
        node.get_logger().info("Success")
    else:
        node.get_logger().error('Service call failed')

    node.get_logger().info("Setting cartesian controller target...")
    if use_pretense:
        node.set_target_cartesian_speed(0.02,0.0,0.0,0.0,False)
    else:
        node.direct_start_ses()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()