import numpy as np
import matplotlib.pyplot as plt


def get_fourier_states(contour, order):
    """
    contour: Kx2 array, [u,v] points
    N: order of Fourier series
    returns: sss feature vector
    """
    #contour = np.array(nodes)   # shape (N, 2)
    N = order #Fourier order
    K = contour.shape[0]
    u = contour[:,0]
    v = contour[:,1]
    
    rho = np.pi * (np.arange(K)) / K  # rho_i = (i-1)pi/K
    
    # Build F matrix for each point
    F = np.zeros((K, 2*N+1))
    for j in range(1, N+1):
        F[:,2*j-2] = np.cos(j * rho)
        F[:,2*j-1] = np.sin(j * rho)
    F[:,-1] = 1  # constant term
    
    # Construct G
    zeros = np.zeros_like(F)
    G = np.block([
        [F, zeros],
        [zeros, F]
    ])
    
    # Stack u and v
    c = np.hstack([u,v])
    
    # Solve linear least squares
    s, _, _, _ = np.linalg.lstsq(G, c, rcond=None)

    approx = (G @ s).reshape(2, contour.shape[0]).T
    
    return s


def get_cartesian_state(contour, nodes=[]):
    """
    target_beam: numpy array of shape (N, 2)
    target_nodes: list or array of indices, e.g. [2, 5]

    returns: numpy array of shape (2 * len(target_nodes),)
    """
    if len(nodes) == 0:
        selected = contour
    else:
        selected = contour[nodes]   # shape = (k, 2)
    # Flatten into a single vector
    return selected.flatten()



def normal_motion(d, l, s, t, p0):
    """Return the (x, y) coordinates of a linear movement (back and forth) at speed s
    of length l and direction d=[dx,dy,dz] as a function of time t."""
    # Normalize direction vector
    d = np.array(d, dtype=float)
    d /= np.linalg.norm(d)

    # Compute total period for a full back-and-forth cycle
    T = 2 * l / s

    # Determine position along the line (triangle wave)
    phase = (t % T) / T  # goes 0→1 over a full cycle
    if phase < 0.5:
        # Forward direction
        dist = 2 * l * phase
    else:
        # Backward direction
        dist = 2 * l * (1 - phase)

    # Compute position
    np_pos = np.array(p0) + d * dist
    #print(np_pos)
    position = [np_pos[0], np_pos[1], np_pos[2]]
    return position


def linear_motion_step(s, d, dt):
    """Return the (x, y) coordinates of a linear movement at speed s
    with direction d=[dx,dy,dz] and time step dt."""
    # Normalize direction vector
    d = np.array(d, dtype=float)
    d /= np.linalg.norm(d)

    # Compute position
    np_pos = d * s * dt 
    #print(np_pos)
    position = [np_pos[0], np_pos[1], np_pos[2]]
    return position


def normal_motion_step(s, f, dt):
    f = np.array(f, dtype=float)
    f_corrected = np.array([-f[1], 0.0, -f[0]], dtype=float)

    # Handle zero-force case to avoid NaN
    n = np.linalg.norm(f_corrected)
    if n < 1:
        # No significant force → default to +y axis
        d = np.array([1.0, 0.0, 0.0])
    else:
        # Normalize force vector → direction of motion
        d = f_corrected / n
    return linear_motion_step(s, d, dt)


def tan_motion_step(s, f, dt):
    f = np.array(f, dtype=float)
    f_corrected = np.array([-f[1], 0.0, -f[0]], dtype=float)

    # Handle zero-force case to avoid NaN
    n = np.linalg.norm(f_corrected)
    if n < 1:
        # No significant force → default to +y axis
        d = np.array([1.0, 0.0, 0.0])
    else:
        # Normalize force vector → direction of motion
        d = f_corrected / n
    d_tan = np.array([-d[2], 0.0, d[0]])
    return linear_motion_step(s, d_tan, dt)


def estimate_deformation_matrix(delta_s_window, delta_r_window):
    """
    delta_sss_window: list of delta_sss vectors, each of size (4N+2,)
    delta_r_window: list of delta_rrr vectors, each of size (3M,)
    returns: QQQ matrix of size (4N+2) x (3M)
    """

    #delta_s_window_np = np.array(delta_s_window, dtype=float)
    #delta_r_window_np = np.array(delta_r_window, dtype=float)

    #delta_s = np.stack(delta_s_window_np, axis=0)  # m x (4N+2)
    #delta_r = np.stack(delta_r_window_np, axis=0)      # m x (3M)

    delta_s = np.asarray(delta_s_window, float)   # shape (m, 4N+2)
    delta_r = np.asarray(delta_r_window, float)   # shape (m, 3M)
    
    # Solve row by row using least squares
    Q = np.zeros((delta_s.shape[1], delta_r.shape[1]))
    for i in range(delta_s.shape[1]):
        sigma_i = delta_s[:,i]  # m
        # Least squares: sigma_i = RRR @ qqq_i
        q_i, _, _, _ = np.linalg.lstsq(delta_r, sigma_i, rcond=None)
        Q[i,:] = q_i
    return Q


def estimate_deformation_matrix_regularized(delta_s_window, delta_r_window, alpha=1e-4, cond_threshold=1e4):
    """
    delta_s_window: list of delta_s vectors, each shape (4N+2,)
    delta_r_window: list of delta_r vectors, each shape (3M,)
    returns: Q matrix of shape (4N+2) x (3M)
    """

    delta_s = np.asarray(delta_s_window, float)   # shape (m, 4N+2)
    delta_r = np.asarray(delta_r_window, float)   # shape (m, 3M)

    m, n_s = delta_s.shape       # m samples, n_s = 4N+2 state dims
    _, n_r = delta_r.shape       # n_r = 3M robot dims

    # Precompute (delta_r^T delta_r + αI)^(-1) delta_r^T  ==> shape (n_r, m)
    RtR = delta_r.T @ delta_r
    A_reg = RtR + alpha * np.eye(n_r)
    Rt = delta_r.T

    # Solve matrix system once:
    #    X = A_reg^{-1} delta_r^T
    # later: q_i = X @ sigma_i
    X = np.linalg.solve(A_reg, Rt)

    # Build Q row by row
    Q = np.zeros((n_s, n_r))
    for i in range(n_s):
        sigma_i = delta_s[:, i]        # shape (m,)
        q_i = X @ sigma_i              # shape (n_r,)
        Q[i, :] = q_i

    # --- Conditioning check ---
    U, S, Vh = np.linalg.svd(Q, full_matrices=False)
    cond_number = (S[0] / S[-1]) if S[-1] > 1e-12 else np.inf
    print("COnd number")
    print(cond_number)
    print("Singular values")
    print(S)
    is_degenerate = cond_number > cond_threshold

    return Q, is_degenerate


def compute_velocity_command(Q, s_current, s_target, tip, eef, lambda_gain=0.1, max_step=1e-5, tip_safe_distance=0.05):
    """
    QQQ: deformation matrix (4N+2)x(3M)
    sss_current: current feature vector
    sss_target: desired feature vector
    lambda_gain: small gain to ensure slow motion
    returns: delta_rrr command for robot
    """
    delta_s = s_target - s_current
    print("### Delta S")
    print(delta_s)
    # Pseudo-inverse
    QQQ_pinv = np.linalg.pinv(Q)
    delta_r = lambda_gain * QQQ_pinv @ delta_s
    
    print("### Delta R")
    print(delta_r)

    # Forbidden tip direction (hard or soft)
    tip_2d = tip[[0, 2]]
    eef_2d  = eef[[0, 2]]
    rod_axis = (tip_2d - eef_2d)
    rod_axis /= np.linalg.norm(rod_axis)
    print(np.linalg.norm(tip_2d - eef_2d))
    if np.linalg.norm(tip_2d - eef_2d) < tip_safe_distance:
        motion_to_tip = np.dot(delta_r, rod_axis)
        print(motion_to_tip)
        if motion_to_tip > 0:
            print(delta_r)
            delta_r -= motion_to_tip * rod_axis  # remove tip component
            print(delta_r)
            print("###")

    norm = np.linalg.norm(delta_r)
    if norm > max_step:
        delta_r = delta_r * (max_step / norm)

    # # Project onto x-z plane for tip constraint
    # tip_2d = tip[[0, 2]]
    # eef_2d = eef[[0, 2]]

    # # Vector pointing away from tip
    # cone_axis = eef_2d - tip_2d
    # dist_tip = np.linalg.norm(cone_axis)
    # if dist_tip > 1e-8:
    #     cone_axis /= dist_tip  # unit vector

    #     # Only apply tip constraint if close to tip
    #     if dist_tip < tip_safe_distance:
    #         cos_theta = np.dot(delta_r, cone_axis) / (np.linalg.norm(delta_r) + 1e-8)

    #         # If motion points into the forbidden cone (toward tip), project it
    #         if cos_theta < 0:  # outside allowed 90° cone
    #             print("Robot close to rod tip")
    #             # Project δr onto plane perpendicular to cone axis
    #             delta_r -= np.dot(delta_r, cone_axis) * cone_axis
    #             # Write back to original delta_r

    # # Step-size limiting
    # norm = np.linalg.norm(delta_r)
    # if norm > max_step:
    #     delta_r = delta_r * (max_step / norm)

    return delta_r

def jacobian_exploration(data, t, last_t, dt, dt_2, R3, dR_cmd, beam_ids, dS, dR, last_S, last_R, m, d, i):
    """
    Docstring for jacobian_exploration
    
    :param data: Mujoco mj.Data
    :param t: current time
    :param last_t: last time at which dS and dR were updated
    :param dt: time step of simulation
    :param dt_2: time step of actuation
    :param R3: 3D position of end effector
    :param beam_ids: IDs of the tracked beam features/nodes
    :param dS: Vector of state (S) differences
    :param dR: Vector of robot end effector position (R) differences
    :param last_S: Last rod state
    :param last_R: Last end effector position
    :param m: Size of dS and dR used to defined the deformation matrix/Jacobian
    :param d: Exploration directions
    :param i: Current exploration iteration
    """
    if (t - last_t) >= dt_2:
        R = np.array([R3[0], R3[2]])
        beam_nodes = []
        for site_i in range(len(beam_ids)):
            node_pos = data.site(beam_ids[site_i]).xpos.copy()
            beam_nodes.append([node_pos[0], node_pos[2]])
        #beam_states = get_fourier_states(contour=np.array(beam_nodes), order=2)
        beam_states = get_cartesian_state(np.array(beam_nodes))
        if len(last_S)==len(beam_states) and len(last_R)==len(R):
            if i%2==0: # Don't save the come back
                dS.append(beam_states - last_S)
                dR.append(R - last_R)
                #print("Append")
            i+=1
        last_S = beam_states
        last_R = R
        
        if i < (m*2):
            d_i = d[i]
            #print(d_i)
            dR_cmd = np.array(linear_motion_step(s=0.015, d=d_i, dt=dt), dtype=float)
        else:
            #print(S)
            #print(R)
            dR_cmd = np.zeros(3)
        last_t = t
    return dR_cmd, dS, dR, last_S, last_R, i, last_t,


#NEW METHODS

import numpy as np

def estimate_deformation_matrix_regularized_(delta_s_window, delta_r_window, alpha=1e-4, cond_threshold=1e6):
    """
    Robust regularized estimator of the deformation mapping W such that:
        S ≈ R @ W
    where:
      - R is delta_r_window (m x n_r)
      - S is delta_s_window (m x n_s)
    Returns:
      W: (n_r x n_s)
      is_degenerate: True if conditioning is bad
    """
    R = np.asarray(delta_r_window, dtype=float)   # (m, n_r)
    S = np.asarray(delta_s_window, dtype=float)   # (m, n_s)

    if R.ndim != 2 or S.ndim != 2:
        raise ValueError("delta_r_window and delta_s_window must be 2D arrays")

    m, n_r = R.shape
    m2, n_s = S.shape
    if m != m2:
        raise ValueError("delta_r and delta_s must have same number of rows (samples)")

    # Tikhonov regularization: W = (R^T R + alpha I)^{-1} R^T S
    RtR = R.T @ R
    A = RtR + alpha * np.eye(n_r)
    RtS = R.T @ S
    W = np.linalg.solve(A, RtS)   # shape (n_r, n_s)

    # conditioning check (on A or W)
    try:
        U, svals, Vt = np.linalg.svd(W, full_matrices=False)
        cond_number = svals[0] / svals[-1] if svals[-1] > 1e-12 else np.inf
    except np.linalg.LinAlgError:
        cond_number = np.inf

    is_degenerate = cond_number > cond_threshold
    return W, is_degenerate


def compute_velocity_command_(W, s_current, s_target, tip, eef, lambda_gain=0.1,
                             max_step=1e-5, tip_safe_distance=0.05, reg=1e-4):
    """
    Given estimated mapping W (n_r x n_s), compute delta_r so that:
        s_target - s_current = delta_s ≈ r @ W
    We solve for r with least squares: W.T @ r^T = delta_s^T -> r = solve(W.T, delta_s^T)
    Return delta_r as a (n_r,) vector.

    W : np.array shape (n_r, n_s)
    s_current, s_target : vectors length n_s
    """
    delta_s = (s_target - s_current).astype(float)   # shape (n_s,)

    # If delta_s is all zeros, return zero
    if np.linalg.norm(delta_s) < 1e-12:
        return np.zeros(W.shape[0])

    # Solve W.T @ r^T = delta_s^T  ->  r^T = lstsq(W.T, delta_s^T)
    # i.e. solve linear system for r in least squares sense
    # Use small Tikhonov regularization by augmenting W.T and delta_s.T
    WT = W.T   # shape (n_s, n_r)
    # Solve WT @ rT = delta_sT  -> rT shape (n_r,)
    # Use lstsq which handles under/overdetermined
    try:
        rT, *_ = np.linalg.lstsq(WT, delta_s.reshape(-1,1), rcond=None)
        r = rT.flatten()
    except np.linalg.LinAlgError:
        # fallback to pseudo-inverse
        r = np.linalg.pinv(W) @ delta_s

    # Apply lambda gain (scale down)
    delta_r = lambda_gain * r

    # TIP SAFETY: remove component moving toward the tip (same idea as you had)
    # Convert tip/eef to 2D projected axis (x,z)
    tip_2d = np.asarray(tip)[[0,2]]
    eef_2d = np.asarray(eef)[[0,2]]
    vec_tip = tip_2d - eef_2d
    dist = np.linalg.norm(vec_tip)
    if dist > 1e-8:
        rod_axis = vec_tip / dist
        if dist < tip_safe_distance:
            # Project delta_r onto robot-space axis corresponding to x-z components if applicable.
            # We need to know which entries of delta_r correspond to x and z movement.
            # In your code you treat delta_r as [dx, dz] (2 dims). So handle that case:
            if delta_r.size >= 2:
                # build a small r_xz using indices 0 and 1 (user must map to correct dims)
                r_xz = np.array([delta_r[0], delta_r[1]])
                motion_to_tip = np.dot(r_xz, rod_axis)
                if motion_to_tip > 0:
                    # subtract component in rod_axis using same mapping
                    r_xz = r_xz - motion_to_tip * rod_axis
                    delta_r[0] = r_xz[0]
                    delta_r[1] = r_xz[1]

    # Step-size limiting
    norm = np.linalg.norm(delta_r)
    if norm > max_step and norm > 0:
        delta_r = delta_r * (max_step / norm)

    return delta_r
