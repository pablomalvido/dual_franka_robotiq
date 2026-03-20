"""
Paper-Ready: Progress-Variable-Driven Ergodic Control for Rod Manipulator

This script produces:
1. MP4 video of the simulation
2. High-quality paper-ready figures

Key innovations:
1. Ergodic-settling gate: ρ(E(t)) = 1/(1 + exp(γ(E(t) - E_ref)))
2. Virtual progress variable s(t): ds/dt = ρ(E(t)), s(0) = 0
3. Temperature θ(s) and memory T(s) parameterized by progress s (not time t)
4. Mixing dynamics: dα/dt = κ_α θ(s)^2 ρ(E(t)), guaranteeing:
   - Asymptotic stability: |dα/dt|/θ(t) = κ_α θ(t) ρ(E) → 0
   - Guaranteed completion: α(s) = ∫ κ_α θ(τ)^2 dτ diverges → α → 1
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3, WrenchStamped
from custom_msgs.msg import SesStarter, MarkerPoseArray, MarkerPose
from std_msgs.msg import Bool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib import rc
import json
import yaml
import os
import time
from ament_index_python.packages import get_package_share_directory

# Enable paper-quality text rendering
rc('font', **{'family': 'sans-serif', 'size': 10})
rc('text', usetex=False)

def load_config(config_name):
    """Load configuration from YAML file.
    
    Looks for config in:
    1. Current working directory
    2. Package config directory
    
    Returns empty dict if not found (defaults will be used).
    """
    # Try current directory first
    if os.path.exists(config_name):
        config_path = config_name
    else:
        # Try package config directory
        try:
            pkg_dir = get_package_share_directory('mujoco_ros')
            config_path = os.path.join(pkg_dir, 'config', config_name)
        except Exception:
            # Fallback to source directory
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', config_name
            )
    
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    else:
        print(f"Config file not found: {config_path}. Using defaults.")
        return {}


class ChainErgodicServoingProgressDriven(Node):
    """
    Bayesian Ergodic Control for a 2-DOF rod manipulator with PROGRESS-VARIABLE-DRIVEN DYNAMICS.
    """

    def __init__(self,
                 dt=0.05,
                 max_steps=1500,
                 num_modes=15,
                 config_file='ses_controller_real_exp6_v3.yaml'):
        
        super().__init__("ses_controller_real")

        # Load configuration from YAML (overrides defaults)
        cfg = load_config(config_file)
        
        self.experiment_name = cfg.get('experiment_name', 'exp3')

        # ROS interfaces to communicate with Mujoco simulation
        self.q_des_pub = self.create_publisher(Vector3, "/cartesian_velocity_controller_ses/target_pos", 10) #Actuation command (Desired x, z position for the robot)
        self.activate_ses_subs = self.create_subscription(SesStarter, "ses/activate", self._activate_ses_cb, 10) #Activate controller and receive initial position to configure the domain
        self.marker_poses_subs = self.create_subscription(MarkerPoseArray, "ses/aruco_poses", self._marker_poses_cb, 10) #Current marker poses
        self.forces_sub = self.create_subscription(WrenchStamped, 'nordbo/wrench',self.read_forces,10) #1
        self.forces = np.zeros(6)
        self.finish_pub = self.create_publisher(Bool, "ses/finish", 10) #Actuation command (Desired x, z position for the robot)


        self.dt = cfg.get('dt', dt)
        self.max_steps = cfg.get('max_steps', max_steps)
        
        # Parameter Bounds relative to the EE initial position
        self.K_MIN_rel = cfg.get('K_MIN_rel', -0.01)  # X
        self.K_MAX_rel = cfg.get('K_MAX_rel', 0.2)
        self.L1_MIN_rel = cfg.get('L1_MIN_rel', -0.2)  # Z
        self.L1_MAX_rel = cfg.get('L1_MAX_rel', 0.01)
        
        # Features on End-Effector
        self.n_feat = cfg.get('n_feat', 1)
        tracked_feat = cfg.get('tracked_feat', [0])
        self.s_feat = np.array(tracked_feat) #Monitored rod nodes indexes (From 0 to number of trackers)

        # Current value of targets, initialization
        self.r_feat = np.zeros((2, len(self.s_feat)))
        self.r_feat_all = np.zeros((2, self.n_feat)) #All the nodes
        
        # Progress-Variable Dynamics
        self.gate_gamma = cfg.get('gate_gamma', 5.0)
        self.gate_E_ref = cfg.get('gate_E_ref', 0.1)
        self.s_progress = 0.0
        
        # Temperature schedule: θ(s) = θ_0 / ln(e + λ_θ s)
        self.theta_0 = cfg.get('theta_0', 5.0)
        self.lambda_theta = cfg.get('lambda_theta', 10.0)
        
        # Memory schedule: T(s) = T_0 * exp(-λ_T s)
        self.T_0 = cfg.get('T_0', 15.0)
        self.lambda_T = cfg.get('lambda_T', 0.1)
        self.T_min = self.dt
        
        # Mixing dynamics: dα/dt = κ_α θ(s)^2 ρ(E)
        self.kappa_alpha = cfg.get('kappa_alpha', 0.03)
        self.alpha_mix = 0.0
        
        self.theta = self.theta_0
        self.T = self.T_0
        
        # Sobolev exponent
        self.s_sobolev = cfg.get('s_sobolev', 1.5)
        
        # Bayesian Parameters
        self.sigma_apriori = cfg.get('sigma_apriori', 0.25)
        self.sigma_meas_norm = cfg.get('sigma_meas_norm', 0.01)
        self.sigma_likelihood = cfg.get('sigma_likelihood', 0.01)
        
        # Grid Setup
        self.N_grid = cfg.get('N_grid', 200)
        
        # Spectral Setup
        num_modes = cfg.get('num_modes', num_modes)
        self.k_idx_list = []
        for k1 in range(num_modes):
            for k2 in range(num_modes):
                self.k_idx_list.append([k1, k2])
        self.k_idx_list = np.array(self.k_idx_list)
        self.num_modes_actual = len(self.k_idx_list)
        
        # Control State
        self.c_k = np.zeros(self.num_modes_actual)
        self.phi_history = []
        
        # History/Logging
        self.trajectory_history = []
        self.metric_history = []
        self.control_history = []
        self.ergodic_energy_history = []
        self.gate_rho_history = []
        
        self.logs = {
            "t": [], "t_real": [], "K": [], "L1": [], "error": [],
            "alpha": [], "s_progress": [], "T": [], "theta": [],
            "E_ergodic": [], "rho_gate": [], "alpha_dot": [], "stability_ratio": [],
            "forces": [], "r_feat": [], "r_targ": [], "Phi_target_iter": [], "Phi_target": []
        }

        self.iter = 0
        self.initialized = False
        self.ses_active = False
        self.base_dir = "/ros2_ws/src/developments/dual_franka_robotiq/mujoco_ros/data/" + self.experiment_name + "/"

    def _activate_ses_cb(self, msg):
        if not self.initialized:
            #Rod target poses
            if not msg.ses_target.poses:
                self.get_logger().warn("Received empty PoseArray")
                return
            # Extract X and Y coordinates
            x_vals = [pose.position.x for pose in msg.ses_target.poses]
            y_vals = [pose.position.y for pose in msg.ses_target.poses] #Note: We use y as z because for the camera z is the depth axis
            # Create 2xN array
            self.r_targ = np.array([x_vals, y_vals])
            self.get_logger().info(f"Updated r_targ with {len(msg.ses_target.poses)} poses")

            #End effector initial position
            self.init_position = np.array([msg.start_position.x, msg.start_position.y, msg.start_position.z])
            self.init_conditions(msg.start_position.x, msg.start_position.z)
            self.t_init = time.time()
        self.ses_active = True

    def _marker_poses_cb(self, msg: MarkerPoseArray):
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y #Note: We use y as z because for the camera z is the depth axis
            marker_id = marker.id
            # Fill all features
            if marker_id < self.n_feat:
                self.r_feat_all[0, marker_id] = x
                self.r_feat_all[1, marker_id] = y
        # Fill r_feat with only selected features
        # self.s_feat contains marker IDs
        for f_idx, marker_id in enumerate(self.s_feat):
            self.r_feat[0, f_idx] = self.r_feat_all[0, marker_id]
            self.r_feat[1, f_idx] = self.r_feat_all[1, marker_id]

    def read_forces(self,msg):
        self.forces=np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])


    def init_conditions(self, X0, Z0):
        """Initialize grid and probability functions after the initial robot position is know. As the domain is relative to the starting position after pre-tense motion"""
        self.initialized = True
        # Parameter Bounds (the domain is a rectangle in the cartesian plane XZ).
        self.X0 = X0 #0.4467193
        self.Z0 = Z0 #0.488
        self.K_MIN = self.X0 + self.K_MIN_rel #X
        self.K_MAX = self.X0 + self.K_MAX_rel
        self.L1_MIN = self.Z0 + self.L1_MIN_rel #Z
        self.L1_MAX = self.Z0 + self.L1_MAX_rel

        print("STARTING SES CONTROLLER")
        print("SES CONTROLLER STARTING CONDITIONS: x="+str(self.X0)+", z="+str(self.Z0))
        print("DOMAIN LIMITS: X=["+str(self.K_MIN)+","+str(self.K_MAX)+"], Z=["+str(self.L1_MIN)+","+str(self.L1_MAX)+"]")
        
        self.K_SPAN = self.K_MAX - self.K_MIN
        self.L_SPAN = self.L1_MAX - self.L1_MIN

        # Grid Setup
        self.n_vec = np.linspace(0, 1, self.N_grid)
        self.KN_GRID, self.LN_GRID = np.meshgrid(self.n_vec, self.n_vec)
        
        self.K_GRID = self.K_MIN + self.KN_GRID * self.K_SPAN
        self.L1_GRID = self.L1_MIN + self.LN_GRID * self.L_SPAN

        # Spectral Setup        
        self._compute_basis_tensors()
        
        # Distributions
        self.use_prior = False
        if self.use_prior:
            self.Phi_Apriori = self._load_prior(999) #In case we want to load the prior obtained from the previous exploration
            self.Phi_GT = self._load_prior(999) #In case we want to load the prior obtained from the previous exploration
        else:
            self.Phi_Apriori = self._compute_prior_uniform() #Uniform distribution as prior
            self.Phi_GT = self._compute_prior_uniform() #self._compute_ground_truth(). JUST REQUIRED TO GENERATE FIGURE
        self.Phi_Accum_Evidence = np.zeros((self.N_grid, self.N_grid))
        self.Phi_Accum_Visits = np.zeros((self.N_grid, self.N_grid)) + 1e-9
        
        # Control State
        self.q_curr = np.array([self.X0, self.Z0]) #Initial position in the Q domain (not normalized) (K=X, L1=Z)

        #Start control loop
        self.timer = self.create_timer(self.dt, self.step)

    def _compute_basis_tensors(self):
        """Precompute cosine basis functions on the grid."""
        self.Basis_Tensor = np.zeros((self.N_grid, self.N_grid, self.num_modes_actual))
        for m in range(self.num_modes_actual):
            k1, k2 = self.k_idx_list[m]
            self.Basis_Tensor[:, :, m] = (np.cos(k1 * np.pi * self.KN_GRID) * 
                                          np.cos(k2 * np.pi * self.LN_GRID))
        self.Basis_Flat = self.Basis_Tensor.reshape(self.N_grid * self.N_grid, self.num_modes_actual)

    def _load_prior(self, index=999):
        if index == 999:
            existing_dirs = [
                int(d) for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d)) and d.isdigit()
            ]
            index = max(existing_dirs) if existing_dirs else 1
        else:
            filename = os.path.join(self.base_dir, str(index), "Phi_Apriori.npy")
            if os.path.exists(filename):
                print(f"Loading distribution from {filename}")
                return np.load(filename)
            else:
                print("No saved distribution found. Using uniform prior.")
                return self._compute_prior_uniform()

    
    def _compute_prior_uniform(self):
        """Compute a uniform distribution in the domain"""
        Phi = np.ones([self.N_grid, self.N_grid], dtype = float)
        Phi /= np.sum(Phi)
        return Phi

    def compute_ergodic_gate(self, ergodic_energy):
        """Compute ergodic-settling gate: ρ(E) = 1/(1 + exp(γ(E - E_ref)))."""
        return 1.0 / (1.0 + np.exp(self.gate_gamma * (ergodic_energy - self.gate_E_ref)))

    def update_progress_and_schedules(self, rho_gate):
        """Update progress variable s and compute θ(s), T(s) schedules."""
        self.s_progress += rho_gate * self.dt
        self.theta = self.theta_0 / np.log(np.e + self.lambda_theta * self.s_progress)
        self.T = self.T_0 * np.exp(-self.lambda_T * self.s_progress)
        self.T = max(self.T, self.T_min)

    def update_alpha(self, rho_gate):
        """Update mixing parameter: dα/dt = κ_α θ(s)^2 ρ(E), saturating at α=1."""
        alpha_dot = self.kappa_alpha * (self.theta ** 2) * rho_gate
        self.alpha_mix += alpha_dot * self.dt
        self.alpha_mix = min(self.alpha_mix, 1.0)
        return alpha_dot

    def compute_trajectory_coeffs(self):
        """Compute time-averaged trajectory coefficients with sliding window T."""
        if len(self.phi_history) == 0:
            return np.zeros(self.num_modes_actual)
        
        # If T is very small, return instantaneous state
        if self.T <= 2.0 * self.dt:
            return self.phi_history[-1]
        
        window = max(1, int(self.T / self.dt))
        window = min(window, len(self.phi_history))
        phi = np.array(self.phi_history[-window:])
        
        if len(phi) == 1:
            return phi[0]
        
        integral = 0.5 * phi[0]
        integral += np.sum(phi[1:-1], axis=0)
        integral += 0.5 * phi[-1]
        integral *= self.dt
        
        return integral / self.T

    def step(self):
        """Execute one control step."""
        if not self.ses_active: #Inactive
            return
        
        self.iter += 1
        self.t_real = time.time() - self.t_init
        # Spectral weighting
        k1_idx = self.k_idx_list[:, 0]
        k2_idx = self.k_idx_list[:, 1]
        eig_vals = (k1_idx**2 + k2_idx**2)
        Lambda_curr = 1.0 / (1.0 + eig_vals)**self.s_sobolev
        
        # Sensing using REAL kinematics
        meas_err_sq_sum = 0.0
        real_dist_sum = 0.0
        for f in range(len(self.s_feat)):
            dist = np.linalg.norm(self.r_feat[:, f] - self.r_targ[:, f]) #How far your current position is from the target position
            meas_err_sq_sum += dist**2
            real_dist_sum += dist
        
        lik_val = 1.0 / (1.0 + meas_err_sq_sum / (2.0 * self.sigma_likelihood**2)) #y(x(t)) in the paper. How likely is to accomplish the task in this x
        
        # Bayesian update
        nk = (self.q_curr[0] - self.K_MIN) / self.K_SPAN
        nk = np.clip(nk, 0, 1)
        nl = (self.q_curr[1] - self.L1_MIN) / self.L_SPAN
        nl = np.clip(nl, 0, 1)
        
        Dist_Sq = (self.KN_GRID - nk)**2 + (self.LN_GRID - nl)**2
        Kernel = np.exp(-Dist_Sq / (2.0 * self.sigma_meas_norm**2)) #Gaussian kernel
        
        self.Phi_Accum_Evidence += Kernel * lik_val
        self.Phi_Accum_Visits += Kernel
        
        # Composite target
        P_Obs_Raw = self.Phi_Accum_Evidence / self.Phi_Accum_Visits
        Sum_Obs = np.sum(P_Obs_Raw)
        
        if Sum_Obs > 0:
            P_Obs_Refined = P_Obs_Raw / Sum_Obs
        else:
            P_Obs_Refined = np.zeros_like(P_Obs_Raw)
        
        Phi_Composite = (1.0 - self.alpha_mix) * self.Phi_Apriori + self.alpha_mix * P_Obs_Refined
        Phi_Composite /= np.sum(Phi_Composite) #Normalize
        
        # Inverse diffusion sharpening
        temperature_norm = self.theta / self.theta_0 #Theta(t) rate
        min_temperature = 0.0001
        temperature_norm = max(temperature_norm, min_temperature)
        
        exponent = 1.0 / temperature_norm
        self.Phi_Sharpened = Phi_Composite ** exponent
        self.Phi_Sharpened /= np.sum(self.Phi_Sharpened) #Normalize
        
        # Projection of Phi in spectral basis (change to freq domain)
        phi_k_targ = (self.Phi_Sharpened.flatten() @ self.Basis_Flat) 
        
        # Ergodic control
        cos_k1 = np.cos(k1_idx * np.pi * nk)
        sin_k1 = np.sin(k1_idx * np.pi * nk)
        cos_k2 = np.cos(k2_idx * np.pi * nl)
        sin_k2 = np.sin(k2_idx * np.pi * nl)
        
        Psi_vec = cos_k1 * cos_k2
        dPsi_dnk = -k1_idx * np.pi * sin_k1 * cos_k2 #These are the gradients in the different dimensions of the domain, k and l (I think)
        dPsi_dnl = -k2_idx * np.pi * cos_k1 * sin_k2
        
        self.phi_history.append(Psi_vec.copy())
        
        c_k_traj = self.compute_trajectory_coeffs()
        
        # Ergodic energy
        ergodic_error = c_k_traj - phi_k_targ
        ergodic_energy = np.sum(Lambda_curr * (ergodic_error ** 2))
        
        # Progress-driven dynamics. SCHEDULING
        rho_gate = self.compute_ergodic_gate(ergodic_energy)
        self.update_progress_and_schedules(rho_gate)
        alpha_dot = self.update_alpha(rho_gate)
        stability_ratio = np.abs(alpha_dot) / (self.theta + 1e-10)
        
        # Control law
        Weights = Lambda_curr * ergodic_error
        u_nk = np.sum(Weights * dPsi_dnk) #Actuation for each dimension, it includes the ergodic error (scaled by lambda that penalizes high frequencies) multiplied by its gradient, in each freq k
        u_nl = np.sum(Weights * dPsi_dnl)
        
        control_gain = 0.05
        u_norm = -control_gain * np.array([u_nk, u_nl])
        
        u_phys = u_norm * np.array([self.K_SPAN, self.L_SPAN]) #Omega domain (normalized) --> Q domain
        u_max = 0.5
        # Magnitude-based clipping
        u_norm = np.linalg.norm(u_phys)
        if u_norm > u_max:
            u_phys = u_phys * (u_max / u_norm)
                
        # Update state (THIS SHOULD GO TO MUJOCO AS THE COMMAND)
        self.q_des = self.q_curr + u_phys * self.dt
        self.q_des[0] = np.clip(self.q_des[0], self.K_MIN, self.K_MAX)
        self.q_des[1] = np.clip(self.q_des[1], self.L1_MIN, self.L1_MAX)

        point_msg = Vector3()
        point_msg.x = self.q_des[0]
        point_msg.y = self.init_position[1] #Y is not controlled, we keep it constant at the initial position (the domain is only in XZ)
        point_msg.z = self.q_des[1]
        self.q_des_pub.publish(point_msg)

        self.q_curr = self.q_des # This shouldn't be like this, but the controller should be subscribed to the robot state. We do it like this as the freq of mujoco is much faster than the freq of the controller
        
        # Logging
        self.trajectory_history.append(self.q_curr.copy())
        self.metric_history.append(real_dist_sum)
        self.control_history.append(u_phys.copy())
        self.ergodic_energy_history.append(ergodic_energy)
        self.gate_rho_history.append(rho_gate)
        
        self.logs["t"].append(self.iter * self.dt)
        self.logs["t_real"].append(self.t_real)
        self.logs["K"].append(self.q_curr[0])
        self.logs["L1"].append(self.q_curr[1])
        self.logs["error"].append(real_dist_sum)
        self.logs["alpha"].append(self.alpha_mix)
        self.logs["s_progress"].append(self.s_progress)
        self.logs["T"].append(self.T)
        self.logs["theta"].append(self.theta)
        self.logs["E_ergodic"].append(ergodic_energy)
        self.logs["rho_gate"].append(rho_gate)
        self.logs["alpha_dot"].append(alpha_dot)
        self.logs["stability_ratio"].append(stability_ratio)
        self.logs["forces"].append(self.forces)
        self.logs["r_feat"].append(self.r_feat_all)
        self.logs["r_targ"].append(self.r_targ)
        if self.iter % 30 == 0:
            Phi_Sharpened_downsampled = self.downsample_phi(self.Phi_Sharpened, k=4) #Downsample for visualization purposes
            self.logs["Phi_target_iter"].append(self.iter)
            self.logs["Phi_target"].append(Phi_Sharpened_downsampled)

        if(self.iter % 10==0):
            print(f"  Step {self.iter} | α={self.alpha_mix:.3f} | E={self.logs['E_ergodic'][-1]:.4f} | q_d=" + str(self.q_des) + f" | Error={self.logs['error'][-1]:.4f}")
        
        if(self.iter >= self.max_steps):
            self.timer.cancel()
            print("Simulation complete!")
            print(f"  Final alpha: {self.alpha_mix:.4f}")
            print(f"  Final error: {self.logs['error'][-1]:.4f}")
            print(f"  Final progress: {self.s_progress:.2f}")
            self.finish_pub.publish(Bool(data=True))
            
            # Create paper-ready figure
            print("\nGenerating paper-ready figure...")
            self.create_path(self.base_dir)
            self.create_paper_figure(self.img_path)
            self.save_logs_to_json(self.json_path)
            self.save_distribution(P_Obs_Refined, self.distribution_path)

    def create_path(self, base_dir):
        # Get all existing numbered folders
        existing_dirs = [
            int(d) for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
        ]

        # Determine next index
        next_index = max(existing_dirs) + 1 if existing_dirs else 1

        # Create new directory
        new_dir = os.path.join(base_dir, str(next_index))
        os.makedirs(new_dir, exist_ok=True)

        # File paths
        self.img_path = os.path.join(new_dir, "progress_ergodic_paper_figure.png")
        self.json_path = os.path.join(new_dir, "progress_ergodic_json.json")
        self.distribution_path = os.path.join(new_dir, "P_Obs_Refined.npy")

    def downsample_phi(self, Phi, k):
        N = Phi.shape[0]
        # Ensure divisible
        assert N % k == 0, "Grid size must be divisible by k"
        new_N = N // k
        # Reshape and sum blocks
        Phi_ds = Phi.reshape(new_N, k, new_N, k).sum(axis=(1, 3))
        # Optional: renormalize (should already sum to 1, but safe)
        Phi_ds /= np.sum(Phi_ds)
        return Phi_ds

    def create_paper_figure(self, img_path):
        """Create high-quality paper-ready figure."""
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
        
        # 1. System Visualization (large, top-left)
        ax_system = fig.add_subplot(gs[0:2, 0:2])
                
        ## Real system
        ax_system.plot(self.r_targ[0], -self.r_targ[1], 'gX', 
                    markersize=12, markeredgewidth=2.5, label='Target Features', zorder=5)
        ax_system.plot(self.r_feat_all[0], -self.r_feat_all[1], 'bo', markersize=8, markeredgewidth=2,
                        fillstyle='none', label='Real Features')
        ax_system.plot(self.r_feat[0], -self.r_feat[1], 'ro', markersize=8, markeredgewidth=2,
                        fillstyle='none', label='Real tracked Features')
        
        x_min, y_min = np.min(self.r_feat_all, axis=1)
        x_max, y_max = np.max(self.r_feat_all, axis=1)
        offset = 0.05
        ax_system.set_xlim([x_min - offset, x_max + offset]) 
        ax_system.set_ylim([-y_max - offset, -y_min + offset]) 
        ax_system.set_aspect('equal')
        ax_system.grid(True, alpha=0.3, linestyle='--')
        ax_system.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax_system.set_xlabel('x [m]', fontsize=11)
        ax_system.set_ylabel('y [m]', fontsize=11)
        ax_system.set_title('(a) System Configuration', fontsize=12, fontweight='bold')
        
        # 2. T, theta, alpha evolution
        ax_params = fig.add_subplot(gs[0, 2])
        
        ax_params.plot(self.logs["t"], self.logs["alpha"], 'k-', lw=2, label=r'$\alpha$')
        ax_params.set_ylabel(r'$\alpha$ (Mixing)', fontsize=10, color='k')
        ax_params.tick_params(axis='y', labelcolor='k')
        ax_params.set_ylim([-0.05, 1.05])
        
        ax_params_twin = ax_params.twinx()
        T_norm = np.array(self.logs["T"]) / self.T_0
        theta_norm = np.array(self.logs["theta"]) / self.theta_0
        ax_params_twin.plot(self.logs["t"], T_norm, 'orange', lw=2, linestyle='--', label=r'$T/T_0$')
        ax_params_twin.plot(self.logs["t"], theta_norm, 'blue', lw=2, linestyle='-.', label=r'$\theta/\theta_0$')
        ax_params_twin.set_ylabel('Normalized T, θ', fontsize=10)
        ax_params_twin.legend(loc='upper right', fontsize=8)
        
        ax_params.set_xlabel('Time [s]', fontsize=10)
        ax_params.set_title('(b) Parameter Evolution', fontsize=11, fontweight='bold')
        ax_params.grid(True, alpha=0.3)
        ax_params.legend(loc='upper left', fontsize=8)
        
        # 3. Ergodic Energy and Servoing Error
        ax_errors = fig.add_subplot(gs[0, 3])
        
        ax_errors.semilogy(self.logs["t"], self.logs["E_ergodic"], 'g-', lw=2, label='Ergodic Energy')
        ax_errors.axhline(y=self.gate_E_ref, color='r', linestyle='--', alpha=0.7, 
                        linewidth=1.5, label=f'$E_{{ref}}$={self.gate_E_ref}')
        ax_errors.set_ylabel('Ergodic Energy (log)', fontsize=10, color='g')
        ax_errors.tick_params(axis='y', labelcolor='g')
        
        ax_errors_twin = ax_errors.twinx()
        ax_errors_twin.plot(self.logs["t"], self.logs["error"], 'r-', lw=2, label='Servoing Error')
        ax_errors_twin.set_ylabel('Servoing Error [m]', fontsize=10, color='r')
        ax_errors_twin.tick_params(axis='y', labelcolor='r')
        
        ax_errors.set_xlabel('Time [s]', fontsize=10)
        ax_errors.set_title('(c) Error Metrics', fontsize=11, fontweight='bold')
        ax_errors.grid(True, alpha=0.3)
        ax_errors.legend(loc='upper left', fontsize=8)
        ax_errors_twin.legend(loc='upper right', fontsize=8)
        
        # 4. Control Actions
        ax_control = fig.add_subplot(gs[1, 2:])
        
        u_array = np.array(self.control_history)
        ax_control.plot(self.logs["t"], u_array[:, 0], 'b-', lw=2, label='$u_K$ (Curvature)')
        ax_control.plot(self.logs["t"], u_array[:, 1], 'r-', lw=2, label='$u_{L1}$ (Length)')
        ax_control.set_xlabel('Time [s]', fontsize=10)
        ax_control.set_ylabel('Control Input', fontsize=10)
        ax_control.set_title('(d) Control Actions', fontsize=11, fontweight='bold')
        ax_control.legend(fontsize=9)
        ax_control.grid(True, alpha=0.3)
        
        # 5. A Priori Distribution
        ax_prior = fig.add_subplot(gs[2, 0])
        
        im1 = ax_prior.contourf(self.KN_GRID, self.LN_GRID, self.Phi_Apriori, 
                                levels=15, cmap='Blues')
        ax_prior.set_xlabel('Normalized K', fontsize=9)
        ax_prior.set_ylabel('Normalized $L_1$', fontsize=9)
        ax_prior.set_title('(e) A Priori Distribution', fontsize=10, fontweight='bold')
        plt.colorbar(im1, ax=ax_prior, fraction=0.046, pad=0.04)
        
        # 6. Ground Truth (Real) Distribution
        ax_gt = fig.add_subplot(gs[2, 1])
        
        im2 = ax_gt.contourf(self.KN_GRID, self.LN_GRID, self.Phi_GT, 
                            levels=15, cmap='Reds')
        ax_gt.set_xlabel('Normalized K', fontsize=9)
        ax_gt.set_ylabel('Normalized $L_1$', fontsize=9)
        ax_gt.set_title('(f) Ground Truth Distribution', fontsize=10, fontweight='bold')
        plt.colorbar(im2, ax=ax_gt, fraction=0.046, pad=0.04)
        
        # 7. Discovered Distribution
        ax_disc = fig.add_subplot(gs[2, 2])
        
        P_Obs_Refined = self.Phi_Accum_Evidence / self.Phi_Accum_Visits
        P_Obs_Refined /= np.sum(P_Obs_Refined)
        
        im3 = ax_disc.contourf(self.KN_GRID, self.LN_GRID, P_Obs_Refined, 
                            levels=15, cmap='Greens')
        ax_disc.set_xlabel('Normalized K', fontsize=9)
        ax_disc.set_ylabel('Normalized $L_1$', fontsize=9)
        ax_disc.set_title('(g) Discovered Distribution', fontsize=10, fontweight='bold')
        plt.colorbar(im3, ax=ax_disc, fraction=0.046, pad=0.04)
        
        # 8. Target with Trajectory
        ax_traj = fig.add_subplot(gs[2, 3])
        

        # Final sharpened target
        Phi_Composite = (1.0 - self.alpha_mix) * self.Phi_Apriori + self.alpha_mix * P_Obs_Refined
        Phi_Composite /= np.sum(Phi_Composite)
        temperature_norm = max(self.theta / self.theta_0, 0.0001)
        exponent = 1.0 / temperature_norm
        Phi_Sharpened = Phi_Composite ** exponent
        Phi_Sharpened /= np.sum(Phi_Sharpened)
        
        im4 = ax_traj.contourf(self.KN_GRID, self.LN_GRID, Phi_Sharpened, 
                            levels=15, cmap='viridis')
        
        # Trajectory
        traj = np.array(self.trajectory_history)
        traj_norm_k = (traj[:, 0] - self.K_MIN) / self.K_SPAN
        traj_norm_l = (traj[:, 1] - self.L1_MIN) / self.L_SPAN
        ax_traj.plot(traj_norm_k, traj_norm_l, 'w-', lw=1, alpha=0.7)
        ax_traj.plot(traj_norm_k[-1], traj_norm_l[-1], 'wo', markersize=8, 
                    markerfacecolor='red', markeredgewidth=2)
        
        ax_traj.set_xlabel('Normalized K', fontsize=9)
        ax_traj.set_ylabel('Normalized $L_1$', fontsize=9)
        ax_traj.set_title('(h) Target + Trajectory', fontsize=10, fontweight='bold')
        plt.colorbar(im4, ax=ax_traj, fraction=0.046, pad=0.04)
        
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()


    def save_logs_to_json(self, filename="/ros2_ws/src/developments/dual_franka_robotiq/mujoco_ros/data/exp3_real/logged_data.json"):
        """
        Save self.logs dictionary into a JSON file.
        Converts numpy types to native Python types for compatibility.
        """

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        # Convert logs
        logs_serializable = {
            key: [convert(v) for v in values]
            for key, values in self.logs.items()
        }

        with open(filename, "w") as f:
            json.dump(logs_serializable, f, indent=4)

        print(f"Logs saved to {filename}")


    def save_distribution(self, P_Obs_Refined, filename="/ros2_ws/src/developments/dual_franka_robotiq/mujoco_ros/data/exp3_real/P_Obs_Refined.npy"):
        """Save discovered distribution."""
        np.save(filename, P_Obs_Refined)
        print(f"Saved distribution to {filename}")


def main():
    rclpy.init()
    ses_node = ChainErgodicServoingProgressDriven()
    print("Experiment 1 real Controller initialized. Waiting for simulation...")
    try:
        rclpy.spin(ses_node)   # <-- runs until node is killed
    except KeyboardInterrupt:
        pass
    finally:
        ses_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    #simulate_and_save()
    main()