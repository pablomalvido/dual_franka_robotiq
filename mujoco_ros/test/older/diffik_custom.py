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
from mujoco_ros import fourier_estimation


# Integration timestep in seconds.
integration_dt: float = 1.0
damping: float = 1e-4
gravity_compensation: bool = True
dt: float = 0.002 #0.002
max_angvel = 0.0
dt_2: float = 0.5

#Beam Target
"""
Position beam joint 1: [0.45  0.256 0.1  ]
Position beam joint 2: [0.45498558 0.256      0.14975082]
Position beam joint 3: [0.46427615 0.256      0.19888009]
Position beam joint 4: [0.47715928 0.256      0.24719184]
Position beam joint 5: [0.4929216  0.256      0.29464233]
Position beam joint 6: [0.51086713 0.256      0.34131093]
Position beam joint 7: [0.53032489 0.256      0.38736954]
Position beam joint 8: [0.55064799 0.256      0.43305292]
Position beam joint 9: [0.57120708 0.256      0.47863059]
Position beam joint 10: [0.59177349 0.256      0.52420496]
"""
target_beam = np.zeros((10,2))
target_beam[0] = np.array([0.45, 0.1], dtype=float)
target_beam[1] = np.array([0.45498558, 0.14975082], dtype=float)
target_beam[2] = np.array([0.46427615, 0.19888009], dtype=float)
target_beam[3] = np.array([0.47715928, 0.24719184], dtype=float)
target_beam[4] = np.array([0.4929216, 0.29464233], dtype=float)
target_beam[5] = np.array([0.51086713, 0.34131093], dtype=float)
target_beam[6] = np.array([0.53032489, 0.38736954], dtype=float)
target_beam[7] = np.array([0.55064799, 0.43305292], dtype=float)
target_beam[8] = np.array([0.57120708, 0.47863059], dtype=float)
target_beam[9] = np.array([0.59177349, 0.52420496], dtype=float)

target_state_beam = fourier_estimation.get_fourier_states(contour=target_beam, order=2)

# =============================
# ROS2 NODE
# =============================
class MujocoRosNode(Node):
    def __init__(self):
        super().__init__("mujoco_ros2_node")

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

        self.beam_pub = self.create_publisher(FloatListList, "beam_state", 10)

        # Received mocap target (default none → use initial position)
        self.mocap_target = None

    def mocap_callback(self, msg: Point):
        self.mocap_target = np.array([msg.x, msg.y, msg.z])

    # def normal_motion(self, t: float, s: float, l: float, d: list[float], p0: list[float]):
    #     self.req_motion.t = t
    #     self.req_motion.s = s
    #     self.req_motion.l = l
    #     self.req_motion.d = d
    #     self.req_motion.p0 = [float(p0[0]), float(p0[1]), float(p0[2])]
    #     return self.cli_normal.call_async(self.req_motion)
    
    # def linear_motion_step(self, s: float, d: list[float], p0: list[float]):
    #     self.req_motion.t = dt
    #     self.req_motion.s = s
    #     #self.req_motion.l = l
    #     self.req_motion.d = d
    #     self.req_motion.p0 = [float(p0[0]), float(p0[1]), float(p0[2])]
    #     return self.cli_linear_step.call_async(self.req_motion)

def go_and_return(vectors):
    result = []
    for v in vectors:
        result.append(v)      # go
        result.append(-v)     # come back
    return np.array(result)

def clockwise_directions(m):
    # angles from 0 to -2π (clockwise)
    angles = np.linspace(0, -np.pi, m, endpoint=False)
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

        target_beam_id = model.site("target_beam").id
        #data.site_xpos[target_beam_id] = np.array([target_beam[7][0], 0.256, target_beam[7][1]])
        #data.site_xmat[target_beam_id] = np.eye(3).reshape(-1)

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
        last_motion_time = time.time()

        initial_force = [0,0,0]
        initialized = False
        pre_tense = False
        pre_tense_force = 4

        record_jacobian = False
        m = 10 #Window size
        vectors = clockwise_directions(m)
        exploration_directions = go_and_return(vectors)
        beam_states_diff = deque(maxlen=m) #m samples, 4 * fourier order + 2 
        robot_position_diff = deque(maxlen=m) #m samples, 2 directions/axes
        last_beam_states = []
        last_eef_pos_R2 = []
        change_position = np.zeros(3)
        exp_i = 0
        explore_jac = False

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
            # if ros_node.mocap_target is not None:
            #     target_pos = ros_node.mocap_target
            # else:
            #     # If no ROS position yet, stay at init position
            #     target_pos = init_pos
            
            #### GET TASK SPACE TARGET ###
            #data.mocap_pos[mocap_id, :3] = init_pos
            forces = data.sensordata[0:3] - initial_force 

            # INITIALIZE
            if not initialized: #Wait some time to initialize gravity and so on
                if (time.time() - init_time) > 2:
                    initial_force = data.sensordata[0:3].copy()
                    print("Initial forces: " + str(initial_force))
                    initialized = True
            
            #"""
            # PRETENSE
            elif not pre_tense:
                if np.linalg.norm(forces) < pre_tense_force:
                    #current_position = data.mocap_pos[mocap_id, 0:3].copy()
                    change_position = np.array(fourier_estimation.linear_motion_step(s=0.015, d=[1.0, 0.0, 0.0], dt=dt), dtype=float)
                    #print(next_position)
                    data.mocap_pos[mocap_id, :3] += change_position
                else:
                    beam_nodes = []
                    for site_i in range(len(beam_sites_id)):
                        node_pos = data.site(beam_sites_id[site_i]).xpos.copy()
                        beam_nodes.append([node_pos[0], node_pos[2]])
                    #last_beam_states = fourier_estimation.get_fourier_states(contour=np.array(beam_nodes), order=2)
                    #eef_pos = data.site(site_id).xpos.copy()
                    #last_robot_position = np.array([eef_pos[0], eef_pos[2]])
                    pre_tense = True
                    explore_jac = True


            # INITIAL JACOBIAN EXPLORATION

            # elif len(beam_states_diff) < m:
            #     if (time.time() - last_motion_time) >= dt_2:
            #         eef_pos = data.site(site_id).xpos.copy()
            #         eef_pos_R2 = np.array([eef_pos[0], eef_pos[2]])
            #         beam_nodes = []
            #         for site_i in range(len(beam_sites_id)):
            #             node_pos = data.site(beam_sites_id[site_i]).xpos.copy()
            #             beam_nodes.append([node_pos[0], node_pos[2]])
            #         beam_states = fourier_estimation.get_fourier_states(contour=np.array(beam_nodes), order=2)
            #         if len(last_beam_states)==len(beam_states) and len(last_eef_pos_R2)==len(eef_pos_R2):
            #             beam_states_diff.append(beam_states - last_beam_states)
            #             robot_position_diff.append(eef_pos_R2 - last_eef_pos_R2)
            #         last_beam_states = beam_states
            #         last_eef_pos_R2 = eef_pos_R2
                    
            #         if len(beam_states_diff) < m:
            #             d_i = exploration_directions[len(beam_states_diff)]
            #             change_position = np.array(fourier_estimation.linear_motion_step(s=0.015, d=d_i, dt=dt), dtype=float)
            #         else:
            #             print(beam_states_diff)
            #             print(robot_position_diff)
            #             change_position = np.zeros(3)
            #         last_motion_time = time.time()
            #     data.mocap_pos[mocap_id, :3] += change_position
            elif explore_jac:
                pos_change, beam_states_diff, robot_position_diff, exp_i = fourier_estimation.jacobian_exploration(data=data, t=time.time(), dt=dt, dt_2=dt_2, eef_pos=data.site(site_id).xpos.copy(), beam_ids=beam_sites_id, S=beam_states_diff, R=robot_position_diff, m=m, d=exploration_directions, i=exp_i)
                data.mocap_pos[mocap_id, :3] += pos_change
                if exp_i >= m:
                    explore_jac = False
                    exp_i = 0

            # GO TO TARGET
            else:
                print("AAAA")
                if (time.time() - last_motion_time) >= dt_2:
                    eef_pos = data.site(site_id).xpos.copy()
                    eef_pos_R2 = np.array([eef_pos[0], eef_pos[2]])
                    beam_nodes = []
                    for site_i in range(len(beam_sites_id)):
                        node_pos = data.site(beam_sites_id[site_i]).xpos.copy()
                        beam_nodes.append([node_pos[0], node_pos[2]])
                    beam_states = fourier_estimation.get_fourier_states(contour=np.array(beam_nodes), order=2)
                    if np.linalg.norm(change_position)>=1e-20: #Do not append if there are no changes
                        beam_states_diff.append(beam_states - last_beam_states)
                        robot_position_diff.append(eef_pos_R2 - last_eef_pos_R2)
                    last_beam_states = beam_states
                    last_eef_pos_R2 = eef_pos_R2

                    Q = fourier_estimation.estimate_deformation_matrix(beam_states_diff, robot_position_diff)
                    delta_r = fourier_estimation.compute_velocity_command(Q, beam_states, target_state_beam, lambda_gain=((dt/dt_2)*0.1))
                    change_position = np.array([delta_r[0], 0.0, delta_r[1]])
                    last_motion_time = time.time()
                data.mocap_pos[mocap_id, :3] += change_position
            #"""

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

            # if record_jacobian:
            #     eef_pos = data.site(site_id).xpos.copy()
            #     robot_position = np.array([eef_pos[0], eef_pos[2]])
            #     beam_nodes = []
            #     for site_i in range(len(beam_sites_id)):
            #         node_pos = data.site(beam_sites_id[site_i]).xpos.copy()
            #         beam_nodes.append([node_pos[0], node_pos[2]])
            #     beam_states = fourier_estimation.get_fourier_states(contour=np.array(beam_nodes), order=2)
            #     beam_states_diff.append(beam_states - last_beam_states)
            #     robot_position_diff.append(robot_position - last_robot_position)
            #     last_beam_states = beam_states
            #     last_robot_position = robot_position                

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
