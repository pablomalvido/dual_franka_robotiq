import rclpy
import numpy as np
from rclpy.node import Node
from custom_msgs.srv import MujocoController


class MujocoMotionServices(Node):

    def __init__(self):
        super().__init__('mujoco_motion_services')
        self.srv = self.create_service(MujocoController, 'normal_motion', self.normal_motion)
        self.get_logger().info("Service server ready. Waiting for requests...")

    def normal_motion(self, request, response):
        """Return the (x, y) coordinates of a linear movement (back and forth) at speed s
        of length l and direction d=[dx,dy,dz] as a function of time t."""
        # Normalize direction vector
        d = np.array(request.d, dtype=float)
        d /= np.linalg.norm(d)

        # Compute total period for a full back-and-forth cycle
        T = 2 * request.l / request.s

        # Determine position along the line (triangle wave)
        phase = (request.t % T) / T  # goes 0→1 over a full cycle
        if phase < 0.5:
            # Forward direction
            dist = 2 * request.l * phase
        else:
            # Backward direction
            dist = 2 * request.l * (1 - phase)

        # Compute position
        np_pos = np.array(request.p0) + d * dist
        print(np_pos)
        response.position = [np_pos[0], np_pos[1], np_pos[2]]
        response.success = True
        print(response)
        return response
    

    def linear_motion(self, request, response):
        """Return the (x, y) coordinates of a linear movement at speed s
        with direction d=[dx,dy,dz] and time step t."""
        # Normalize direction vector
        d = np.array(request.d, dtype=float)
        d /= np.linalg.norm(d)

        # Compute position
        np_pos = np.array(request.p0) + d * request.s * request.t 
        #print(np_pos)
        response.position = [np_pos[0], np_pos[1], np_pos[2]]
        response.success = True
        #print(response)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MujocoMotionServices()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()