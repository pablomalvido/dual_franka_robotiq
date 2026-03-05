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

#####ADAPT TO NOT USE ROS
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


def estimate_deformation_matrix(delta_s_window, delta_r_window):
    """
    delta_sss_window: list of delta_sss vectors, each of size (4N+2,)
    delta_r_window: list of delta_rrr vectors, each of size (3M,)
    returns: QQQ matrix of size (4N+2) x (3M)
    """

    delta_s_window_np = np.array(delta_s_window, dtype=float)
    delta_r_window_np = np.array(delta_r_window, dtype=float)

    delta_s = np.stack(delta_s_window_np, axis=0)  # m x (4N+2)
    delta_r = np.stack(delta_r_window_np, axis=0)      # m x (3M)
    
    # Solve row by row using least squares
    Q = np.zeros((delta_s.shape[1], delta_r.shape[1]))
    for i in range(delta_s.shape[1]):
        sigma_i = delta_s[:,i]  # m
        # Least squares: sigma_i = RRR @ qqq_i
        q_i, _, _, _ = np.linalg.lstsq(delta_r, sigma_i, rcond=None)
        Q[i,:] = q_i
    return Q


def compute_velocity_command(Q, s_current, s_target, lambda_gain=0.1):
    """
    QQQ: deformation matrix (4N+2)x(3M)
    sss_current: current feature vector
    sss_target: desired feature vector
    lambda_gain: small gain to ensure slow motion
    returns: delta_rrr command for robot
    """
    delta_s = s_target - s_current
    # Pseudo-inverse
    QQQ_pinv = np.linalg.pinv(Q)
    delta_r = lambda_gain * QQQ_pinv @ delta_s
    return delta_r

def jacobian_exploration(data, t, dt, dt_2, eef_pos, beam_ids, S, R, m, d, i):
    print(beam_ids)
    if (t - last_motion_time) >= dt_2:
        eef_pos_R2 = np.array([eef_pos[0], eef_pos[2]])
        beam_nodes = []
        for site_i in range(len(beam_ids)):
            node_pos = data.site(beam_ids[site_i]).xpos.copy()
            beam_nodes.append([node_pos[0], node_pos[2]])
        beam_states = get_fourier_states(contour=np.array(beam_nodes), order=2)
        if len(last_beam_states)==len(beam_states) and len(last_eef_pos_R2)==len(eef_pos_R2):
            S.append(beam_states - last_beam_states)
            R.append(eef_pos_R2 - last_eef_pos_R2)
            i+=1
        last_beam_states = beam_states
        last_eef_pos_R2 = eef_pos_R2
        
        if i < m:
            d_i = d[i]
            change_position = np.array(linear_motion_step(s=0.015, d=d_i, dt=dt), dtype=float)
        else:
            #print(S)
            #print(R)
            change_position = np.zeros(3)
        last_motion_time = t
    return change_position, S, R, i
