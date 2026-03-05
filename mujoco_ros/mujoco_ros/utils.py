import numpy as np
import matplotlib.pyplot as plt

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