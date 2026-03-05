import sys
import rclpy
import numpy as np
import time
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, Pose, Vector3, WrenchStamped
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped

class MotionOrchestrator(Node):

    def __init__(self):
        super().__init__('motion_orchestrator')
        self.write_data = False
        self.log=[]
        self.log_pos=[]
        self.file_path = "/ros2_ws/src/developments/dual_franka_robotiq/python_scripts_pkg/files/force_log.txt"
        self.file_path_pos = "/ros2_ws/src/developments/dual_franka_robotiq/python_scripts_pkg/files/pos_log.txt"
        #self.file_path = "../files/force_log.txt"

        # Create TF buffer and listener to get the EE position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame = 'fr3_link0'
        self.ee_frame = 'fr3_link8'

        self.cli_switch_controller = self.create_client(SwitchController, 'controller_manager/switch_controller')
        self.cartesian_speed_controller_target_pub = self.create_publisher(Vector3, '/cartesian_acceleration_controller/target_speed', 10)
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
    
    def set_target_cartesian_speed(self, vx, vy, vz, t, timer=True):
        msg = Vector3()
        msg.x = vx
        msg.y = vy
        msg.z = vz
        self.cartesian_speed_controller_target_pub.publish(msg)
        if timer:
            self.start_time=time.time()
            self.timer=self.create_timer(t, self.timer_callback)
            self.timer_tf = self.create_timer(0.08, self.tf_callback)  # 12.5 Hz # Timer to query transform periodically. Tf does not update faster (robot_state_publisher rate is slow just for visualization)
            self.write_data = True

    def timer_callback(self):
        # Stop the timer so it runs only once
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        self.set_target_cartesian_speed(0.0,0.0,0.0,0.0,False)
        self.write_data = False
        self.write_txt()

        #Switch controller again
        stop_controllers = ['cartesian_acceleration_controller']#['cartesian_velocity_example_controller']
        start_controllers = []
        self.get_logger().info("Switching controllers...")
        #future4 = self.switch_controller(start_c=start_controllers, stop_c=stop_controllers)
        self.switch_controller_req.start_controllers = start_controllers
        self.switch_controller_req.stop_controllers = stop_controllers
        self.switch_controller_req.strictness = SwitchController.Request.STRICT
        self.switch_controller_req.timeout = Duration(sec=2, nanosec=0)
        future = self.cli_switch_controller.call_async(self.switch_controller_req)
        future.add_done_callback(self.switch_done_1_cb)

    
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
        self.get_logger().info("Switching controllers...")
        self.switch_controller_req.start_controllers = start_controllers
        self.switch_controller_req.stop_controllers = stop_controllers
        self.switch_controller_req.strictness = SwitchController.Request.STRICT
        self.switch_controller_req.timeout = Duration(sec=2, nanosec=0)
        self.cli_switch_controller.call_async(self.switch_controller_req)   
        

    def read_forces(self,msg):
        Force=np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        if self.write_data:
            self.log.append({'fx': msg.wrench.force.x, 'fy': msg.wrench.force.y, 'fz': msg.wrench.force.z, 'F': np.linalg.norm(Force), 't': (time.time()-self.start_time)})

    def write_txt(self):
        self.get_logger().info('Saving logged data')
        with open(self.file_path, "w") as f:
            for entry in self.log:
                f.write(f"{entry['fx']}\t{entry['fy']}\t{entry['fz']}\t{entry['F']}\t{entry['t']}\n")
        with open(self.file_path_pos, "w") as f:
            for entry in self.log_pos:
                f.write(f"{entry['x']}\t{entry['y']}\t{entry['z']}\t{entry['t']}\n")

    def tf_callback(self):
        try:
            # Lookup transform from base -> end effector
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time()
            )

            translation = transform.transform.translation
            #rotation = transform.transform.rotation

            self.log_pos.append({'x': translation.x, 'y': translation.y, 'z': translation.z, 't': (time.time()-self.start_time)})

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Transform not available: {e}")

def main(args=None):
    rclpy.init(args=args)

    node = MotionOrchestrator()

    #config = "ready_kaleel"

    stop_controllers = ['fr3_arm_controller']
    start_controllers = ['cartesian_acceleration_controller']#['cartesian_velocity_example_controller']

    node.get_logger().info("Switching controllers...")
    future3 = node.switch_controller(start_c=start_controllers, stop_c=stop_controllers)
    rclpy.spin_until_future_complete(node, future3)
    if future3.result() is not None:
        node.get_logger().info("Success")
    else:
        node.get_logger().error('Service call failed')

    node.get_logger().info("Setting cartesian controller target...")
    #node.set_target_cartesian_speed(0.05,0.0,0.0,5.0,True)
    node.set_target_cartesian_speed(0.05,0.0,0.0,0.3,True)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()