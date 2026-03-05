import rclpy
import numpy as np
from rclpy.node import Node
from custom_msgs.msg import FloatListList, FloatList
import matplotlib.pyplot as plt


class OnlineJacobian(Node):

    def __init__(self):
        super().__init__('online_jacobian')
        self.subscription = self.create_subscription(
            FloatListList,
            'beam_state',
            self.listener_callback,
            10)

        # Create figure once
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.raw_plot, = self.ax.plot([], [], 'bo-', markersize=4, label="Raw contour")
        self.approx_plot, = self.ax.plot([], [], 'r-', linewidth=2, label="Fourier approx")
        self.ax.legend()
        self.ax.set_title("Contour vs Fourier Approximation")
        self.ax.set_xlabel("u")
        self.ax.set_ylabel("v")
        self.ax.set_xlim(0.4, 0.6)


    def listener_callback(self, msg):
        nodes = []
        for row in msg.rows:
            nodes.append(row.data)
        contour = np.array(nodes)   # shape (N, 2)
        N = 2 #Fourier order

        s, G = self.compute_fourier_features(contour, N)
                # Compute Fourier approximation: (2K,) → reshape to (K,2)

        approx = (G @ s).reshape(2, contour.shape[0]).T

        # ---- Update Plot ----
        self.update_plot(contour, approx)


    def update_plot(self, contour, approx):
        u = contour[:,0]
        v = contour[:,1]

        ua = approx[:,0]
        va = approx[:,1]

        self.raw_plot.set_data(u, v)
        self.approx_plot.set_data(ua, va)

        # Auto-rescale
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def compute_fourier_features(self, contour, N):
        """
        contour: Kx2 array, [u,v] points
        N: order of Fourier series
        returns: sss feature vector
        """
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
        return s, G
    

def main(args=None):
    rclpy.init(args=args)
    node = OnlineJacobian()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()