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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib import rc

# Enable paper-quality text rendering
rc('font', **{'family': 'sans-serif', 'size': 10})
rc('text', usetex=False)


class RodErgodicServoingProgressDriven:
    """
    Bayesian Ergodic Control for a 2-DOF rod manipulator with PROGRESS-VARIABLE-DRIVEN DYNAMICS.
    """

    def __init__(self,
                 domain_size=1.0,
                 dt=0.05,
                 T_sim=1500,
                 num_modes=15):
        
        self.L = domain_size
        self.dt = dt
        self.T_sim = T_sim
        
        # Parameter Bounds (the domain is a rectangle in the cartesian plane XY). Make this bounds relative to the initial current position
        self.X0 = 0.0 
        self.Z0 = 0.0 
        self.X0 = 0.4467193 
        self.Z0 = 0.488
        self.K_MIN = self.X0 - 0.15 #X
        self.K_MAX = self.X0 + 0.15
        self.L1_MIN = self.Z0 - 0.15 #Z
        self.L1_MAX = self.Z0 + 0.15
        
        self.K_SPAN = self.K_MAX - self.K_MIN
        self.L_SPAN = self.L1_MAX - self.L1_MIN
        
        # Features on End-Effector
        self.s_feat = np.array([0])

        # Real targets
        self.r_targ = np.array([
            [self.X0 + 0.1],
            [self.Z0 - 0.1]
        ])
        
        # Progress-Variable Dynamics
        self.gate_gamma = 5.0
        self.gate_E_ref = 0.25
        self.s_progress = 0.0
        
        # Temperature schedule: θ(s) = θ_0 / ln(e + λ_θ s)
        self.theta_0 = 5.0
        self.lambda_theta = 10.0
        
        # Memory schedule: T(s) = T_0 * exp(-λ_T s)
        self.T_0 = 15.0
        self.lambda_T = 0.1
        self.T_min = self.dt
        
        # Mixing dynamics: dα/dt = κ_α θ(s)^2 ρ(E)
        self.kappa_alpha = 0.03
        self.alpha_mix = 0.0
        
        self.theta = self.theta_0
        self.T = self.T_0
        
        # Sobolev exponent
        self.s_sobolev = 1.1
        
        # Bayesian Parameters
        self.sigma_apriori = 0.25
        self.sigma_meas_norm = 0.01
        self.sigma_likelihood = 0.01
        
        # Grid Setup
        self.N_grid = 200
        self.n_vec = np.linspace(0, 1, self.N_grid)
        self.KN_GRID, self.LN_GRID = np.meshgrid(self.n_vec, self.n_vec)
        
        self.K_GRID = self.K_MIN + self.KN_GRID * self.K_SPAN
        self.L1_GRID = self.L1_MIN + self.LN_GRID * self.L_SPAN
        
        # Spectral Setup
        self.num_modes = num_modes
        self.k_idx_list = []
        for k1 in range(num_modes):
            for k2 in range(num_modes):
                self.k_idx_list.append([k1, k2])
        self.k_idx_list = np.array(self.k_idx_list)
        self.num_modes_actual = len(self.k_idx_list)
        
        self._compute_basis_tensors()
        
        # Distributions
        self.Phi_Apriori = self._compute_prior_uniform() #Uniform distribution
        self.Phi_Accum_Evidence = np.zeros((self.N_grid, self.N_grid))
        self.Phi_Accum_Visits = np.zeros((self.N_grid, self.N_grid)) + 1e-9
        self.Phi_GT = self._compute_prior() #self._compute_ground_truth()
        
        # Control State
        self.q_curr = np.array([self.X0, self.Z0]) #Initial position in the Q domain (not normalized) (K=X, L1=Z)
        self.c_k = np.zeros(self.num_modes_actual)
        self.phi_history = []
        
        # History/Logging
        self.trajectory_history = []
        self.metric_history = []
        self.control_history = []
        self.ergodic_energy_history = []
        self.gate_rho_history = []
        
        self.logs = {
            "t": [], "K": [], "L1": [], "error": [],
            "alpha": [], "s_progress": [], "T": [], "theta": [],
            "E_ergodic": [], "rho_gate": [], "alpha_dot": [], "stability_ratio": []
        }

    def _compute_basis_tensors(self):
        """Precompute cosine basis functions on the grid."""
        self.Basis_Tensor = np.zeros((self.N_grid, self.N_grid, self.num_modes_actual))
        for m in range(self.num_modes_actual):
            k1, k2 = self.k_idx_list[m]
            self.Basis_Tensor[:, :, m] = (np.cos(k1 * np.pi * self.KN_GRID) * 
                                          np.cos(k2 * np.pi * self.LN_GRID))
        self.Basis_Flat = self.Basis_Tensor.reshape(self.N_grid * self.N_grid, self.num_modes_actual)

    def _compute_prior(self):
        """Compute a priori distribution using MODEL kinematics."""
        Phi = np.zeros((self.N_grid, self.N_grid))
        for i in range(self.N_grid):
            for j in range(self.N_grid):
                q_test = np.array([self.K_GRID[i, j], self.L1_GRID[i, j]])
                err_sq_sum = 0.0
                for f in range(len(self.s_feat)):
                    r = self.kinematics_forward_model_cart(q_test, self.s_feat[f])
                    err_sq_sum += np.sum((r - self.r_targ[:, f])**2)
                Phi[i, j] = np.exp(-err_sq_sum / (2 * self.sigma_apriori**2))
        Phi /= np.sum(Phi)
        return Phi
    
    def _compute_prior_uniform(self):
        """Compute a uniform distribution in the domain"""
        Phi = np.ones([self.N_grid, self.N_grid], dtype = float)
        Phi /= np.sum(Phi)
        return Phi

    def _compute_ground_truth(self):
        """Compute ground truth distribution using REAL kinematics."""
        Phi = np.zeros((self.N_grid, self.N_grid))
        for i in range(self.N_grid):
            for j in range(self.N_grid):
                q_test = np.array([self.K_GRID[i, j], self.L1_GRID[i, j]])
                err_sq = 0.0
                for f in range(len(self.s_feat)):
                    r = self.kinematics_forward_real_cart(q_test, self.s_feat[f])
                    err_sq += np.sum((r - self.r_targ[:, f])**2)
                Phi[i, j] = np.exp(-err_sq / (2 * 0.05**2))
        Phi /= np.sum(Phi)
        return Phi

    @staticmethod
    def kinematics_forward_model(q, s):
        """MODEL forward kinematics: Clean constant-curvature rod (2-DOF)."""
        k_in = q[0]
        l1_in = q[1]
        epsk = 1e-10
        
        if s <= l1_in:
            if np.abs(k_in) < epsk:
                x = 0.0
                y = s
            else:
                x = (np.cos(k_in * s) - 1.0) / k_in
                y = np.sin(k_in * s) / k_in
        else:
            sprime = s - l1_in
            A = k_in * l1_in
            if np.abs(k_in) < epsk:
                x = 0.0
                y = s
            else:
                Xa = (np.cos(A) - 1.0) / k_in
                Ya = np.sin(A) / k_in
                theta_e = np.pi / 2.0 + A
                x = Xa + sprime * np.cos(theta_e)
                y = Ya + sprime * np.sin(theta_e)
        
        return np.array([x, y])

    @staticmethod
    def kinematics_forward_model_cart(q, s):
        return np.array([q[0], q[1]])

    @staticmethod
    def kinematics_forward_real(q, s):
        """REAL forward kinematics: Piece-wise rod with modeling errors."""
        k_in = q[0]
        l1_in = q[1]
        epsk = 1e-3
        
        # Base compliance and offsets
        k_compliance = 0.8
        k_actual = k_in * k_compliance
        l1_offset = 0.1
        l1_actual = l1_in + l1_offset
        
        s_norm = s / (1.0 + epsk)
        
        # Multi-frequency zigzag components
        zigzag1 = 0.08 * np.sin(5.0 * np.pi * s_norm)
        zigzag2 = 0.05 * np.sin(11.0 * np.pi * s_norm)
        zigzag3 = 0.03 * np.cos(7.0 * np.pi * s_norm)
        zigzag4 = 0.04 * np.sin(13.0 * np.pi * s_norm + 1.2)
        
        asymmetry = 0.06 * (s_norm**2) * np.sign(k_in)
        k_variation = 1.0 + 0.15 * np.sin(4.0 * np.pi * s_norm) + 0.1 * np.cos(9.0 * np.pi * s_norm)
        
        if s <= l1_actual:
            s_normalized = s / (l1_actual + epsk)
            k_first = k_actual * 1.12 * k_variation
            
            if np.abs(k_first) < epsk:
                x = 0.0
                y = s
            else:
                x = (np.cos(k_first * s) - 1.0) / k_first
                y = np.sin(k_first * s) / k_first
                
                x += zigzag1 + zigzag2 + asymmetry
                y += zigzag3 * 0.5
                
                x += 0.025 * s_normalized * np.sin(6.0 * np.pi * s_normalized)
        else:
            sprime = s - l1_actual
            sprime_norm = sprime / (1.0 - l1_actual + epsk)
            
            s_base_norm = l1_actual / (l1_actual + epsk)
            k_first = k_actual * 1.12 * (1.0 + 0.15 * np.sin(4.0 * np.pi * s_base_norm) 
                                         + 0.1 * np.cos(9.0 * np.pi * s_base_norm))
            A = k_first * l1_actual
            
            if np.abs(k_first) < epsk:
                x = 0.0
                y = s
            else:
                Xa = (np.cos(A) - 1.0) / k_first
                Ya = np.sin(A) / k_first
                
                zigzag1_base = 0.08 * np.sin(5.0 * np.pi * s_base_norm)
                zigzag2_base = 0.05 * np.sin(11.0 * np.pi * s_base_norm)
                asymmetry_base = 0.06 * (s_base_norm**2) * np.sign(k_in)
                
                Xa += zigzag1_base + zigzag2_base + asymmetry_base
                Ya += 0.03 * np.cos(7.0 * np.pi * s_base_norm) * 0.5
                
                theta_e = np.pi / 2.0 + A
                theta_offset = 0.12 + 0.08 * np.sin(3.0 * np.pi * sprime_norm)
                theta_actual = theta_e + theta_offset
                
                x = Xa + sprime * np.cos(theta_actual)
                y = Ya + sprime * np.sin(theta_actual)
                
                x += zigzag1 + zigzag2 * 1.5 + asymmetry * 1.2
                y += zigzag3 + zigzag4
                
                y -= 0.04 * sprime * (1.0 + sprime_norm)
                x += 0.03 * sprime_norm * np.cos(4.0 * np.pi * sprime_norm)
        
        return np.array([x, y])

    @staticmethod
    def kinematics_forward_real_cart(q, s):
        return np.array([q[0], q[1]])

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

    def step(self, t):
        """Execute one control step."""
        
        # Spectral weighting
        k1_idx = self.k_idx_list[:, 0]
        k2_idx = self.k_idx_list[:, 1]
        eig_vals = (k1_idx**2 + k2_idx**2)
        Lambda_curr = 1.0 / (1.0 + eig_vals)**self.s_sobolev
        
        # Sensing using REAL kinematics
        meas_err_sq_sum = 0.0
        real_dist_sum = 0.0
        for f in range(len(self.s_feat)):
            r_feat = self.kinematics_forward_real_cart(self.q_curr, self.s_feat[f])
            dist = np.linalg.norm(r_feat - self.r_targ[:, f]) #How far your current position is from the target position
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
        Phi_Sharpened = Phi_Composite ** exponent
        Phi_Sharpened /= np.sum(Phi_Sharpened) #Normalize
        
        # Projection of Phi in spectral basis (change to freq domain)
        phi_k_targ = (Phi_Sharpened.flatten() @ self.Basis_Flat) 
        
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
        
        # Update state (THIS SHOULD GO TO MUJOCO AND RECEIVE ITS FEEDBACK)
        self.q_curr = self.q_curr + u_phys * self.dt
        self.q_curr[0] = np.clip(self.q_curr[0], self.K_MIN, self.K_MAX)
        self.q_curr[1] = np.clip(self.q_curr[1], self.L1_MIN, self.L1_MAX)
        
        # Inject disturbance at 83% through simulation (to test rejection)
        if t == 2500:
            disturbance = np.array([1.5, 0.15])  # Significant disturbance in both DOF
            self.q_curr = self.q_curr + disturbance
            self.q_curr[0] = np.clip(self.q_curr[0], self.K_MIN, self.K_MAX)
            self.q_curr[1] = np.clip(self.q_curr[1], self.L1_MIN, self.L1_MAX)
            print(f"\n  [DISTURBANCE INJECTED at t={t*self.dt:.1f}s: ΔK={disturbance[0]:.2f}, ΔL1={disturbance[1]:.2f}]\n")
        
        # Logging
        self.trajectory_history.append(self.q_curr.copy())
        self.metric_history.append(real_dist_sum)
        self.control_history.append(u_phys.copy())
        self.ergodic_energy_history.append(ergodic_energy)
        self.gate_rho_history.append(rho_gate)
        
        self.logs["t"].append(t * self.dt)
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
        
        return Phi_Sharpened, P_Obs_Refined


def simulate_and_save(generate_video=False):
    """Run simulation and save video + paper-ready figures.
    
    Args:
        generate_video: If True, generates animation video (slow). Default: False
    """
    
    sim = RodErgodicServoingProgressDriven(T_sim=3000, num_modes=15)
    
    print("=" * 70)
    print("Progress-Variable-Driven Ergodic Control - Paper-Ready Version")
    print("=" * 70)
    print("Running simulation...")
    
    # Store frames for video
    frames_data = []
    
    # Run simulation
    for t in range(sim.T_sim):
        Phi_Target, P_Obs_Refined = sim.step(t)
        
        # Store frame data every 10 steps (only if video generation is enabled)
        if generate_video and t % 10 == 0:
            frames_data.append({
                't': sim.logs["t"][-1],
                'q_curr': sim.q_curr.copy(),
                'Phi_Target': Phi_Target.copy(),
                'P_Obs_Refined': P_Obs_Refined.copy(),
                'trajectory': np.array(sim.trajectory_history).copy()
            })
        
        if t % 300 == 0:
            print(f"  Step {t}/{sim.T_sim} | α={sim.alpha_mix:.3f} | E={sim.logs['E_ergodic'][-1]:.4f}")
    
    print("Simulation complete!")
    print(f"  Final alpha: {sim.alpha_mix:.4f}")
    print(f"  Final error: {sim.logs['error'][-1]:.4f}")
    print(f"  Final progress: {sim.s_progress:.2f}")
    
    # Create paper-ready figure
    print("\nGenerating paper-ready figure...")
    create_paper_figure(sim)
    
    # Create video (optional)
    if generate_video:
        print("\nGenerating video...")
        create_video(sim, frames_data)
    else:
        print("\nVideo generation skipped (set generate_video=True to enable)")
    
    print("\n" + "=" * 70)
    print("Output files created:")
    print("  - progress_ergodic_paper_figure.png")
    writer_name = 'ffmpeg'
    if generate_video:
        try:
            if writer_name == 'ffmpeg':
                print("  - progress_ergodic_simulation.mp4")
            else:
                print("  - progress_ergodic_simulation.gif")
        except NameError:
            print("  (Video generation skipped - no writer available)")
    print("=" * 70)


def create_paper_figure(sim):
    """Create high-quality paper-ready figure."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    # 1. System Visualization (large, top-left)
    ax_system = fig.add_subplot(gs[0:2, 0:2])
    
    s_vec = np.linspace(0, sim.L, 100)
    
    # Real system
    rx_real = np.array([sim.kinematics_forward_real_cart(sim.q_curr, s) for s in s_vec])
    
    # Model prediction
    rx_model = np.array([sim.kinematics_forward_model_cart(sim.q_curr, s) for s in s_vec])
    
    ax_system.plot(rx_real[:, 0], rx_real[:, 1], 'b-', lw=3, label='Real System', alpha=0.9)
    ax_system.plot(rx_model[:, 0], rx_model[:, 1], 'r--', lw=2, label='Model (Prior)', alpha=0.7)
    ax_system.plot(sim.r_targ[0], sim.r_targ[1], 'gX', 
                  markersize=12, markeredgewidth=2.5, label='Target Features', zorder=5)
    
    # Real features
    for f in range(len(sim.s_feat)):
        r_f = sim.kinematics_forward_real_cart(sim.q_curr, sim.s_feat[f])
        ax_system.plot(r_f[0], r_f[1], 'mo', markersize=8, markeredgewidth=2,
                      fillstyle='none', label='Real Features' if f == 0 else '')
    
    # Model features
    for f in range(len(sim.s_feat)):
        r_f = sim.kinematics_forward_model_cart(sim.q_curr, sim.s_feat[f])
        ax_system.plot(r_f[0], r_f[1], 'rs', markersize=6, alpha=0.6,
                      label='Model Features' if f == 0 else '')
    
    ax_system.set_xlim([-0.2, 1.5])
    ax_system.set_ylim([-1.0, 1.0])
    ax_system.set_aspect('equal')
    ax_system.grid(True, alpha=0.3, linestyle='--')
    ax_system.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_system.set_xlabel('x [m]', fontsize=11)
    ax_system.set_ylabel('y [m]', fontsize=11)
    ax_system.set_title('(a) System Configuration', fontsize=12, fontweight='bold')
    
    # 2. T, theta, alpha evolution
    ax_params = fig.add_subplot(gs[0, 2])
    
    ax_params.plot(sim.logs["t"], sim.logs["alpha"], 'k-', lw=2, label=r'$\alpha$')
    ax_params.set_ylabel(r'$\alpha$ (Mixing)', fontsize=10, color='k')
    ax_params.tick_params(axis='y', labelcolor='k')
    ax_params.set_ylim([-0.05, 1.05])
    
    ax_params_twin = ax_params.twinx()
    T_norm = np.array(sim.logs["T"]) / sim.T_0
    theta_norm = np.array(sim.logs["theta"]) / sim.theta_0
    ax_params_twin.plot(sim.logs["t"], T_norm, 'orange', lw=2, linestyle='--', label=r'$T/T_0$')
    ax_params_twin.plot(sim.logs["t"], theta_norm, 'blue', lw=2, linestyle='-.', label=r'$\theta/\theta_0$')
    ax_params_twin.set_ylabel('Normalized T, θ', fontsize=10)
    ax_params_twin.legend(loc='upper right', fontsize=8)
    
    ax_params.set_xlabel('Time [s]', fontsize=10)
    ax_params.set_title('(b) Parameter Evolution', fontsize=11, fontweight='bold')
    ax_params.grid(True, alpha=0.3)
    ax_params.legend(loc='upper left', fontsize=8)
    
    # 3. Ergodic Energy and Servoing Error
    ax_errors = fig.add_subplot(gs[0, 3])
    
    ax_errors.semilogy(sim.logs["t"], sim.logs["E_ergodic"], 'g-', lw=2, label='Ergodic Energy')
    ax_errors.axhline(y=sim.gate_E_ref, color='r', linestyle='--', alpha=0.7, 
                     linewidth=1.5, label=f'$E_{{ref}}$={sim.gate_E_ref}')
    ax_errors.set_ylabel('Ergodic Energy (log)', fontsize=10, color='g')
    ax_errors.tick_params(axis='y', labelcolor='g')
    
    ax_errors_twin = ax_errors.twinx()
    ax_errors_twin.plot(sim.logs["t"], sim.logs["error"], 'r-', lw=2, label='Servoing Error')
    ax_errors_twin.set_ylabel('Servoing Error [m]', fontsize=10, color='r')
    ax_errors_twin.tick_params(axis='y', labelcolor='r')
    
    ax_errors.set_xlabel('Time [s]', fontsize=10)
    ax_errors.set_title('(c) Error Metrics', fontsize=11, fontweight='bold')
    ax_errors.grid(True, alpha=0.3)
    ax_errors.legend(loc='upper left', fontsize=8)
    ax_errors_twin.legend(loc='upper right', fontsize=8)
    
    # 4. Control Actions
    ax_control = fig.add_subplot(gs[1, 2:])
    
    u_array = np.array(sim.control_history)
    ax_control.plot(sim.logs["t"], u_array[:, 0], 'b-', lw=2, label='$u_K$ (Curvature)')
    ax_control.plot(sim.logs["t"], u_array[:, 1], 'r-', lw=2, label='$u_{L1}$ (Length)')
    ax_control.set_xlabel('Time [s]', fontsize=10)
    ax_control.set_ylabel('Control Input', fontsize=10)
    ax_control.set_title('(d) Control Actions', fontsize=11, fontweight='bold')
    ax_control.legend(fontsize=9)
    ax_control.grid(True, alpha=0.3)
    
    # 5. A Priori Distribution
    ax_prior = fig.add_subplot(gs[2, 0])
    
    im1 = ax_prior.contourf(sim.KN_GRID, sim.LN_GRID, sim.Phi_Apriori, 
                            levels=15, cmap='Blues')
    ax_prior.set_xlabel('Normalized K', fontsize=9)
    ax_prior.set_ylabel('Normalized $L_1$', fontsize=9)
    ax_prior.set_title('(e) A Priori Distribution', fontsize=10, fontweight='bold')
    plt.colorbar(im1, ax=ax_prior, fraction=0.046, pad=0.04)
    
    # 6. Ground Truth (Real) Distribution
    ax_gt = fig.add_subplot(gs[2, 1])
    
    im2 = ax_gt.contourf(sim.KN_GRID, sim.LN_GRID, sim.Phi_GT, 
                         levels=15, cmap='Reds')
    ax_gt.set_xlabel('Normalized K', fontsize=9)
    ax_gt.set_ylabel('Normalized $L_1$', fontsize=9)
    ax_gt.set_title('(f) Ground Truth Distribution', fontsize=10, fontweight='bold')
    plt.colorbar(im2, ax=ax_gt, fraction=0.046, pad=0.04)
    
    # 7. Discovered Distribution
    ax_disc = fig.add_subplot(gs[2, 2])
    
    P_Obs_Refined = sim.Phi_Accum_Evidence / sim.Phi_Accum_Visits
    P_Obs_Refined /= np.sum(P_Obs_Refined)
    
    im3 = ax_disc.contourf(sim.KN_GRID, sim.LN_GRID, P_Obs_Refined, 
                           levels=15, cmap='Greens')
    ax_disc.set_xlabel('Normalized K', fontsize=9)
    ax_disc.set_ylabel('Normalized $L_1$', fontsize=9)
    ax_disc.set_title('(g) Discovered Distribution', fontsize=10, fontweight='bold')
    plt.colorbar(im3, ax=ax_disc, fraction=0.046, pad=0.04)
    
    # 8. Target with Trajectory
    ax_traj = fig.add_subplot(gs[2, 3])
    

    # Final sharpened target
    Phi_Composite = (1.0 - sim.alpha_mix) * sim.Phi_Apriori + sim.alpha_mix * P_Obs_Refined
    Phi_Composite /= np.sum(Phi_Composite)
    temperature_norm = max(sim.theta / sim.theta_0, 0.0001)
    exponent = 1.0 / temperature_norm
    Phi_Sharpened = Phi_Composite ** exponent
    Phi_Sharpened /= np.sum(Phi_Sharpened)
    
    im4 = ax_traj.contourf(sim.KN_GRID, sim.LN_GRID, Phi_Sharpened, 
                           levels=15, cmap='viridis')
    
    # Trajectory
    traj = np.array(sim.trajectory_history)
    traj_norm_k = (traj[:, 0] - sim.K_MIN) / sim.K_SPAN
    traj_norm_l = (traj[:, 1] - sim.L1_MIN) / sim.L_SPAN
    ax_traj.plot(traj_norm_k, traj_norm_l, 'w-', lw=1, alpha=0.7)
    ax_traj.plot(traj_norm_k[-1], traj_norm_l[-1], 'wo', markersize=8, 
                markerfacecolor='red', markeredgewidth=2)
    
    ax_traj.set_xlabel('Normalized K', fontsize=9)
    ax_traj.set_ylabel('Normalized $L_1$', fontsize=9)
    ax_traj.set_title('(h) Target + Trajectory', fontsize=10, fontweight='bold')
    plt.colorbar(im4, ax=ax_traj, fraction=0.046, pad=0.04)
    
    plt.savefig('progress_ergodic_paper_figure.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_video(sim, frames_data):
    """Create MP4 video of the simulation."""
    
    # Try to find available video writer
    try:
        # Try ffmpeg first
        Writer = animation.writers['ffmpeg']
        writer_name = 'ffmpeg'
    except RuntimeError:
        try:
            # Fall back to pillow for GIF/MP4
            Writer = animation.writers['pillow']
            writer_name = 'pillow'
            print("  Note: ffmpeg not found, using pillow writer (may produce larger files)")
        except RuntimeError:
            print("  Warning: No video writer available (ffmpeg or pillow)")
            print("  Skipping video generation. Install ffmpeg to enable video output.")
            return
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax_system = fig.add_subplot(gs[:, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_params = fig.add_subplot(gs[0, 2])
    ax_errors = fig.add_subplot(gs[1, 1])
    ax_control = fig.add_subplot(gs[1, 2])
    
    def init():
        return []
    
    def animate(frame_idx):
        if frame_idx >= len(frames_data):
            return []
        
        frame = frames_data[frame_idx]
        t = frame['t']
        q_curr = frame['q_curr']
        Phi_Target = frame['Phi_Target']
        traj = frame['trajectory']
        
        # Clear axes
        for ax in [ax_system, ax_phase, ax_params, ax_errors, ax_control]:
            ax.clear()
        
        # System visualization
        s_vec = np.linspace(0, sim.L, 100)
        rx_real = np.array([sim.kinematics_forward_real_cart(q_curr, s) for s in s_vec])
        rx_model = np.array([sim.kinematics_forward_model_cart(q_curr, s) for s in s_vec])
        
        ax_system.plot(rx_real[:, 0], rx_real[:, 1], 'b-', lw=2.5, label='Real', alpha=0.9)
        ax_system.plot(rx_model[:, 0], rx_model[:, 1], 'r--', lw=1.5, label='Model', alpha=0.7)
        ax_system.plot(sim.r_targ[0], sim.r_targ[1], 'gX', markersize=10, markeredgewidth=2)
        
        ax_system.set_xlim([-0.2, 1.5])
        ax_system.set_ylim([-1.0, 1.0])
        ax_system.set_aspect('equal')
        ax_system.grid(True, alpha=0.3)
        ax_system.legend(loc='upper left', fontsize=8)
        ax_system.set_title(f'System (t={t:.1f}s)')
        
        # Phase space
        ax_phase.contourf(sim.KN_GRID, sim.LN_GRID, Phi_Target, levels=15, cmap='viridis')
        if len(traj) > 0:
            traj_norm_k = (traj[:, 0] - sim.K_MIN) / sim.K_SPAN
            traj_norm_l = (traj[:, 1] - sim.L1_MIN) / sim.L_SPAN
            ax_phase.plot(traj_norm_k, traj_norm_l, 'w-', lw=0.5, alpha=0.7)
            ax_phase.plot(traj_norm_k[-1], traj_norm_l[-1], 'wo', markersize=6, markerfacecolor='red')
        ax_phase.set_xlabel('Norm K')
        ax_phase.set_ylabel('Norm L1')
        ax_phase.set_title('Target Distribution')
        
        # Parameters
        idx = min(frame_idx * 10, len(sim.logs["t"]) - 1)
        ax_params.plot(sim.logs["t"][:idx], sim.logs["alpha"][:idx], 'k-', lw=2, label='α')
        ax_params.set_ylim([-0.1, 1.1])
        ax_params.set_ylabel('α')
        ax_params.set_xlabel('Time [s]')
        ax_params.set_title('Mixing Parameter')
        ax_params.legend()
        ax_params.grid(True, alpha=0.3)
        
        # Errors
        ax_errors.semilogy(sim.logs["t"][:idx], sim.logs["E_ergodic"][:idx], 'g-', lw=2)
        ax_errors.axhline(y=sim.gate_E_ref, color='r', linestyle='--', alpha=0.7)
        ax_errors.set_xlabel('Time [s]')
        ax_errors.set_ylabel('Ergodic Energy')
        ax_errors.set_title('Convergence')
        ax_errors.grid(True, alpha=0.3)
        
        # Control
        if idx > 0:
            u_array = np.array(sim.control_history[:idx])
            ax_control.plot(sim.logs["t"][:idx], u_array[:, 0], 'b-', lw=2, label='u_K')
            ax_control.plot(sim.logs["t"][:idx], u_array[:, 1], 'r-', lw=2, label='u_L1')
            ax_control.set_xlabel('Time [s]')
            ax_control.set_ylabel('Control')
            ax_control.set_title('Control Actions')
            ax_control.legend()
            ax_control.grid(True, alpha=0.3)
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                   frames=len(frames_data), interval=50, blit=True)
    
    if writer_name == 'ffmpeg':
        writer = Writer(fps=20, metadata=dict(artist='Progress-Driven Ergodic Control'), bitrate=1800)
        anim.save('progress_ergodic_simulation.mp4', writer=writer)
    else:  # pillow
        writer = Writer(fps=20)
        anim.save('progress_ergodic_simulation.gif', writer=writer)
        print("  Saved as GIF format (ffmpeg not available for MP4)")
    
    plt.close()


if __name__ == "__main__":
    simulate_and_save()
