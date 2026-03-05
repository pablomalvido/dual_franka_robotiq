import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
import pyrealsense2 as rs

class ArucoPublisher(Node):

    def __init__(self):
        super().__init__('aruco_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'aruco_pose', 10)

        # RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()

        # Marker size in meters
        self.marker_size = 0.04

        self.timer = self.create_timer(0.03, self.process_frame)

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(intrinsics.coeffs)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            for i, corner in enumerate(corners):

                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, self.marker_size, camera_matrix, dist_coeffs)

                # Draw axis
                cv2.drawFrameAxes(
                    color_image, camera_matrix, dist_coeffs,
                    rvec[0][0], tvec[0][0], 0.03)

                # Publish pose
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "camera_link"

                pose_msg.pose.position.x = float(tvec[0][0][0])
                pose_msg.pose.position.y = float(tvec[0][0][1])
                pose_msg.pose.position.z = float(tvec[0][0][2])

                rot_matrix, _ = cv2.Rodrigues(rvec[0][0])
                qw = np.sqrt(1 + rot_matrix[0][0] +
                             rot_matrix[1][1] +
                             rot_matrix[2][2]) / 2
                qx = (rot_matrix[2][1] - rot_matrix[1][2]) / (4 * qw)
                qy = (rot_matrix[0][2] - rot_matrix[2][0]) / (4 * qw)
                qz = (rot_matrix[1][0] - rot_matrix[0][1]) / (4 * qw)

                pose_msg.pose.orientation.x = float(qx)
                pose_msg.pose.orientation.y = float(qy)
                pose_msg.pose.orientation.z = float(qz)
                pose_msg.pose.orientation.w = float(qw)

                self.publisher_.publish(pose_msg)

                # Overlay XYZ text
                position_text = f"ID:{ids[i][0]} X:{tvec[0][0][0]:.2f} Y:{tvec[0][0][1]:.2f} Z:{tvec[0][0][2]:.2f}"
                corner_point = tuple(corner[0][0].astype(int))
                cv2.putText(color_image, position_text,
                            corner_point,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

        cv2.imshow("Aruco Tracking", color_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()