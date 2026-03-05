#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from custom_msgs.srv import MujocoController
from custom_msgs.msg import FloatListList, FloatList
from collections import deque
from fourier_estimation import *


# Integration timestep in seconds.
integration_dt: float = 1.0
damping: float = 1e-4
gravity_compensation: bool = True
dt: float = 0.01 #0.002
max_angvel = 0.0


# =============================
# ROS2 NODE
# =============================
class MujocoRosNode(Node):
    def __init__(self):
        super().__init__("mujoco_ros2_node")

        # Camera publisher
        self.cam_pub = self.create_publisher(Image, "mujoco_camera", 10)
        self.bridge = CvBridge()

        self.cli_normal = self.create_client(MujocoController, 'normal_motion')
        while not self.cli_normal.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service normal force not available, waiting...')
        self.req_motion = MujocoController.Request()

        self.cli_linear_step = self.create_client(MujocoController, 'linear_motion_step')
        while not self.cli_linear_step.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service linear motion step not available, waiting...')
        self.cli_linear_step = MujocoController.Request()

        # Mocap subscriber
        self.subscription = self.create_subscription(
            Point,
            "mocap_target",
            self.mocap_callback,
            10,
        )

        self.beam_pub = self.create_publisher(FloatListList, "beam_state", 10)

        # Received mocap target (default none → use initial position)
        self.mocap_target = None

    def mocap_callback(self, msg: Point):
        self.mocap_target = np.array([msg.x, msg.y, msg.z])

    def normal_motion(self, t: float, s: float, l: float, d: list[float], p0: list[float]):
        self.req_motion.t = t
        self.req_motion.s = s
        self.req_motion.l = l
        self.req_motion.d = d
        self.req_motion.p0 = [float(p0[0]), float(p0[1]), float(p0[2])]
        return self.cli_normal.call_async(self.req_motion)
    
    def linear_motion_step(self, s: float, d: list[float], p0: list[float]):
        self.req_motion.t = dt
        self.req_motion.s = s
        #self.req_motion.l = l
        self.req_motion.d = d
        self.req_motion.p0 = [float(p0[0]), float(p0[1]), float(p0[2])]
        return self.cli_linear_step.call_async(self.req_motion)


def clockwise_directions(m):
    # angles from 0 to -2π (clockwise)
    angles = np.linspace(0, -2*np.pi, m, endpoint=False)
    
    x = np.cos(angles)
    y = np.zeros(m)
    z = np.sin(angles)
    
    return np.stack((x, y, z), axis=1)   # shape (m, 3)
    

# =============================
# MAIN MUJOCO + ROS2 LOOP
# =============================
def main():

    # Start ROS
    rclpy.init()
    ros_node = MujocoRosNode()

    # Load the MuJoCo model
    assert mujoco.__version__ >= "3.1.0", "Please use MuJoCo 3.1.0 or later"
    model = mujoco.MjModel.from_xml_path("/home/rosdev/ros2_ws/src/mujoco_ros/universal_robots_ur5e/scene_v2.xml")
    data = mujoco.MjData(model)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    cam_id = model.camera("robot_camera").id
    renderer = mujoco.Renderer(model, width=320, height=240)

    model.opt.timestep = dt

    site_id = model.site("attachment_site").id

    body_names = [
        "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link",
    ]
    body_ids = [model.body(name).id for name in body_names]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    key_id = model.key("home3").id
    mocap_id = model.body("target").mocapid[0]

    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:

        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        mujoco.mj_forward(model, data)

        init_pos = data.site(site_id).xpos.copy()
        init_rot = data.site(site_id).xmat.copy()
        data.mocap_pos[mocap_id, :3] = init_pos
        mujoco.mju_mat2Quat(site_quat, init_rot)
        data.mocap_quat[mocap_id, :] = site_quat

        print("Initial site position:", init_pos)

        beam_sites_id = []
        n_sites = 10
        for i in range(1, n_sites+1):
            beam_sites_id.append(model.site("site"+str(i)).id)

        # camera timing
        framerate = 2
        frame_i = 1
        init_time = time.time()
        frame_slow = 1
        slow_cycle_freq = 5

        initial_force = [0,0,0]
        initialized = False
        pre_tense = False
        pre_tense_force = 3

        record_jacobian = False
        m = 10 #Window size
        exploration_directions = clockwise_directions(m)
        beam_states_diff = deque(maxlen=m) #m samples, 4 * fourier order + 2 
        robot_position_diff = deque(maxlen=m) #m samples, 2 directions/axes
        last_beam_states = []
        last_robot_position = []

        # Structure:
        # 1. Pre-tense: Linear motion until certain force
        # 2. Initial jacobian exploration: 
        # 2a. Call services to move tangencial and normal
        # 2b. Record both action and beam state changes (encode state service)
        # 3. Get next robot action:
        # 3a. Calculate jacobian for certain window
        # 3b. Control and provide eef velocity
        # 4.a Execute computed motion command
        # 4.b Update robot action and beam status changes list

        while viewer.is_running():

            # Spin ROS callbacks
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            # =====================
            # CAMERA → ROS2 PUBLISH
            # =====================

            if (time.time() - init_time) > (frame_slow / slow_cycle_freq):
                beam_state_msg = FloatListList()
                for site_i in range(len(beam_sites_id)):
                    node = FloatList()
                    node_pos = data.site(beam_sites_id[site_i]).xpos.copy()
                    node.data = [node_pos[0], node_pos[2]]
                    beam_state_msg.rows.append(node)
                ros_node.beam_pub.publish(beam_state_msg)
                frame_slow += 1

            if (time.time() - init_time) > (frame_i / framerate):

                # renderer.update_scene(data, camera=cam_id)
                # frame = renderer.render()  # RGB uint8

                # # OpenCV display (optional)
                # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # cv2.imshow("MuJoCo Camera", frame_bgr)
                # cv2.waitKey(1)

                # # Publish to ROS2
                # img_msg = ros_node.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
                # ros_node.cam_pub.publish(img_msg)

                frame_i += 1

            # =====================
            # CONTROL TARGET FROM ROS2
            # =====================
            if ros_node.mocap_target is not None:
                target_pos = ros_node.mocap_target
            else:
                # If no ROS position yet, stay at init position
                target_pos = init_pos
            
            #### GET TASK SPACE TARGET ###
            data.mocap_pos[mocap_id, :3] = target_pos
            forces = data.sensordata[0:3] - initial_force 

            # INITIALIZE
            if not initialized: #Wait some time to initialize gravity and so on
                if (time.time() - init_time) > 0.5:
                    initial_force = data.sensordata[0:3].copy()
                    print("Initial forces: " + str(initial_force))
                    initialized = True

            # PRETENSE
            elif not pre_tense:
                if np.linalg.norm(forces) < pre_tense_force:
                    future = ros_node.linear_motion_step(s=0.015, d=[1.0, 0.0, 0.0], p0=data.site(site_id).xpos.copy())
                    rclpy.spin_until_future_complete(ros_node, future)
                    if future.result() is not None:
                        result = future.result()
                        #print(result)
                        data.mocap_pos[mocap_id, :3] = np.array(result.position, dtype=float)
                    else:
                        ros_node.get_logger().error('Service call failed')
                
                else:
                    beam_nodes = []
                    for site_i in range(len(beam_sites_id)):
                        node_pos = data.site(beam_sites_id[site_i]).xpos.copy()
                        beam_nodes.append([node_pos[0], node_pos[2]])
                    beam_states = get_fourier_states(contour=np.array(beam_nodes), order=2)
                    last_robot_position = data.site(site_id).xpos.copy()
                    pre_tense = True


            # INITIAL JACOBIAN EXPLORATION
            elif len(beam_states_diff) < m:
                record_jacobian = True
                d_i = exploration_directions[len(beam_states_diff)]
                future = ros_node.linear_motion_step(s=0.015, d=d_i, p0=data.site(site_id).xpos.copy())
                rclpy.spin_until_future_complete(ros_node, future)
                if future.result() is not None:
                    result = future.result()
                    data.mocap_pos[mocap_id, :3] = np.array(result.position, dtype=float)
                else:
                    ros_node.get_logger().error('Service call failed')

            # GO TO TARGET
            else:
                pass

            # -------------------------
            # CONTROLLER
            # -------------------------
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            mujoco.mj_step(model, data)
            viewer.sync()

            if record_jacobian:
                robot_position = data.site(site_id).xpos.copy()
                beam_nodes = []
                for site_i in range(len(beam_sites_id)):
                    node_pos = data.site(beam_sites_id[site_i]).xpos.copy()
                    beam_nodes.append([node_pos[0], node_pos[2]])
                beam_states = get_fourier_states(contour=np.array(beam_nodes), order=2)
                beam_states_diff.append(beam_states - last_beam_states)
                robot_position_diff.append(robot_position - last_robot_position)
                last_beam_states = beam_states
                last_robot_position = robot_position                

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
