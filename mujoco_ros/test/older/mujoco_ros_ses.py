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
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from custom_msgs.srv import MujocoController
from custom_msgs.msg import FloatListList, FloatList
from collections import deque
from mujoco_ros import fourier_estimation
from mujoco_ros import utils


# Integration timestep in seconds.
integration_dt: float = 1.0
damping: float = 1e-4
gravity_compensation: bool = True
dt: float = 0.002 #0.002
max_angvel = 0.0
dt_2: float = 0.4


# =============================
# ROS2 NODE
# =============================
class MujocoRosNode(Node):
    def __init__(self):
        super().__init__("mujoco_ros_ses")

        # # Camera publisher
        # self.cam_pub = self.create_publisher(Image, "mujoco_camera", 10)
        # self.bridge = CvBridge()

        # # Mocap subscriber
        # self.subscription = self.create_subscription(
        #     Point,
        #     "mocap_target",
        #     self.mocap_callback,
        #     10,
        # )

        self.ee_pos_pub = self.create_publisher(Point, "ee_pos", 10)
        self.ses_activate_pub = self.create_publisher(Bool, "ses/activate", 10)

        self.ee_target_subs = self.create_subscription(Point, "ses/x_d", self.ee_target_callback, 10)

        # Received mocap target (default none → use initial position)
        self.ee_target = None
        self.count=0

    def ee_target_callback(self, msg: Point):
        self.count+=1
        if self.count % 10 == 0:
            print(self.count)
        self.ee_target = np.array([msg.x, msg.y, msg.z])
    

# =============================
# MAIN MUJOCO + ROS2 LOOP
# =============================
def main():

    # Start ROS
    rclpy.init()
    ros_node = MujocoRosNode()

    # Load the MuJoCo model
    assert mujoco.__version__ >= "3.1.0", "Please use MuJoCo 3.1.0 or later"
    model = mujoco.MjModel.from_xml_path("/home/rosdev/ros2_ws/src/mujoco_ros/universal_robots_ur5e/scene_empty.xml")
    data = mujoco.MjData(model)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

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

        # camera timing
        framerate = 2
        init_time = time.time()
        frame_slow = 1
        slow_cycle_freq = 5
        last_motion_time = time.time()

        initial_force = [0,0,0]
        initialized = False
        pre_tense = False
        pre_tense_force = 3

        record_jacobian = False
        last_beam_states = []
        last_eef_pos_R2 = []
        change_position = np.zeros(3)
        t=time.time()

        # Structure:
        # 1. Pre-tense: Linear motion until certain force
        # 2. 

        while viewer.is_running():

            # Spin ROS callbacks
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            forces = data.sensordata[0:3] - initial_force 

            # ee_pos = data.site(site_id).xpos.copy()
            ee_pos = data.mocap_pos[mocap_id].copy()
            point_msg = Point()
            point_msg.x = ee_pos[0]
            point_msg.y = ee_pos[1]
            point_msg.z = ee_pos[2]
            ros_node.ee_pos_pub.publish(point_msg)
            x_d = ee_pos

            # INITIALIZE
            if not initialized: #Wait some time to initialize gravity and so on
                if (time.time() - init_time) > 2:
                    initial_force = data.sensordata[0:3].copy()
                    print("Initial forces: " + str(initial_force))
                    initialized = True

            # PRETENSE
            elif not pre_tense:
                if data.mocap_pos[mocap_id][0] < 0.4467193:
                    #current_position = data.mocap_pos[mocap_id, 0:3].copy()
                    change_position = np.array(utils.linear_motion_step(s=0.015, d=[1.0, 0.0, 0.0], dt=dt), dtype=float)
                    #print(next_position)
                    data.mocap_pos[mocap_id, :3] += change_position
                    x_d = data.mocap_pos[mocap_id].copy()
                else:
                    pre_tense = True

            else:
                if ros_node.ee_target is None:
                    bool_msg = Bool()
                    bool_msg.data = True
                    ros_node.ses_activate_pub.publish(bool_msg) #Activate SES controller
                else:
                    data.mocap_pos[mocap_id][0] = ros_node.ee_target[0] #X #Updated from subscriber to the controller
                    data.mocap_pos[mocap_id][2] = ros_node.ee_target[2] #Z

            # -------------------------
            # CONTROLLER: data.site is the EEF current position, which tries to follow the mocap_pos (goal)
            # -------------------------
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            # print("\n")
            # print(data.mocap_pos[mocap_id])
            # print(data.site(site_id).xpos)

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            J_ur = jac[:,dof_ids]
            #dq is not really a joint velocity here, because the error is not a cartesian velocity either. 
            #dq is more like the incremental step of how much q has to change for the next iteration, as 
            #based on how the error is calculated, it is suppose to be corrected in one iteration.
            dq_ur = J_ur.T @ np.linalg.solve(J_ur @ J_ur.T + diag, error) 
            #dq_ur = np.linalg.solve(J_ur, error) 

            if max_angvel > 0:
                dq_abs_max = np.abs(dq_ur).max()
                if dq_abs_max > max_angvel:
                    dq_ur *= max_angvel / dq_abs_max

            q = data.qpos.copy()
            dq = np.zeros(model.nv)
            dq[dof_ids] = dq_ur #All the joints of the model (including not only the robot)
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            np.clip(q, *model.jnt_range.T, out=q)

            data.ctrl[actuator_ids] = q[dof_ids]
            # print(time.time()-t)
            # t=time.time()

            mujoco.mj_step(model, data)
            viewer.sync()                

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
