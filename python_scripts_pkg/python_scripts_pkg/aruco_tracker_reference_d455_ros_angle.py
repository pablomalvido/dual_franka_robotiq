import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray, Pose
from custom_msgs.msg import SesStarterAngle, MarkerPoseArray, MarkerPose
from std_msgs.msg import Float64
import cv2
import numpy as np
import pyrealsense2 as rs

class ArucoPublisher(Node):

    def __init__(self):
        super().__init__('aruco_publisher')

        self.marker_pub = self.create_publisher(Float64, 'ses/aruco_angle', 10)

        self.subscription = self.create_subscription(
            SesStarterAngle,
            "ses/activate",
            self.ref_poses_callback,
            10)

        self.latest_target_poses = None

        # ✅ USER-DEFINED TARGET ANGLE
        self.target_angle_deg = 0.0

        self.distance_threshold = 0.05
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()

        self.marker_size = 0.04

        self.timer = self.create_timer(0.03, self.process_frame)

    def ref_poses_callback(self, msg):
        self.target_angle_deg = msg.ses_target
        self.get_logger().info("Received new target poses")

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

        current_angle_deg = None  # ✅ store computed angle

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            marker_positions = []

            for i, corner in enumerate(corners):

                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, self.marker_size, camera_matrix, dist_coeffs)
                
                marker_positions.append((
                    ids[i][0],
                    float(tvec[0][0][0]),
                    float(tvec[0][0][1]),
                    float(tvec[0][0][2])
                ))

                cv2.drawFrameAxes(
                    color_image, camera_matrix, dist_coeffs,
                    rvec[0][0], tvec[0][0], 0.03)

                position_text = f"ID:{ids[i][0]} X:{tvec[0][0][0]:.2f} Y:{tvec[0][0][1]:.2f} Z:{tvec[0][0][2]:.2f}"
                corner_point = tuple(corner[0][0].astype(int))
                cv2.putText(color_image, position_text,
                            corner_point,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

            # ✅ COMPUTE ANGLE (ID 2 → ID 1, XY plane only)
            marker_dict = {m[0]: m for m in marker_positions}

            if 1 in marker_dict and 2 in marker_dict:
                x1, y1 = marker_dict[1][1], marker_dict[1][2]
                x2, y2 = marker_dict[2][1], marker_dict[2][2]

                dx = x1 - x2
                dy = y1 - y2

                angle_rad = np.arctan2(dy, dx)
                current_angle_deg = -np.degrees(angle_rad)
                msg = Float64()
                msg.data = current_angle_deg
                self.marker_pub.publish(msg)

            # Publish markers (unchanged)
            # marker_array_msg = MarkerPoseArray()
            # marker_array_msg.header.stamp = self.get_clock().now().to_msg()
            # marker_array_msg.header.frame_id = "camera_frame"

            # for marker in marker_positions:
            #     marker_id, x, y, z = marker
            #     marker_msg = MarkerPose()
            #     marker_msg.id = int(marker_id)
            #     marker_msg.pose.position.x = x
            #     marker_msg.pose.position.y = y
            #     marker_msg.pose.position.z = z
            #     marker_msg.pose.orientation.w = 1.0
            #     marker_array_msg.markers.append(marker_msg)

            #self.marker_pub.publish(marker_array_msg)

            # ❌ REMOVE target circles (deleted cv2.circle)

        # # ✅ DRAW WHITE INFO BOX
        # box_x, box_y = 20, 20
        # box_w, box_h = 360, 80

        # Get image dimensions
        img_h, img_w, _ = color_image.shape

        box_w, box_h = 360, 80
        margin = 20

        # ✅ Top-right corner placement
        box_x = img_w - box_w - margin
        box_y = margin

        # White background
        cv2.rectangle(color_image,
                      (box_x, box_y),
                      (box_x + box_w, box_y + box_h),
                      (255, 255, 255), -1)

        # Black border
        cv2.rectangle(color_image,
                      (box_x, box_y),
                      (box_x + box_w, box_y + box_h),
                      (0, 0, 0), 2)

        # Target angle text
        cv2.putText(color_image,
                    f"Target angle: {self.target_angle_deg:.2f}",
                    (box_x + 10, box_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)

        # Current angle text
        if current_angle_deg is not None:
            current_text = f"Current angle: {current_angle_deg:.2f}"
        else:
            current_text = "Current angle: N/A"

        cv2.putText(color_image,
                    current_text,
                    (box_x + 10, box_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)

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