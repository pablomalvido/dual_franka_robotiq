import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray, Pose
from custom_msgs.msg import SesStarterAngle, MarkerPoseArray, MarkerPose
from std_msgs.msg import Int32
import cv2
import numpy as np
import pyrealsense2 as rs

class ArucoPublisher(Node):

    def __init__(self):
        super().__init__('aruco_publisher')

        self.marker_pub = self.create_publisher(Int32, 'ses/aruco_count', 10)

        # self.subscription = self.create_subscription(
        #     SesStarterAngle,
        #     "ses/activate",
        #     self.ref_poses_callback,
        #     10)

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

    # def ref_poses_callback(self, msg):
    #     #self.target_angle_deg = msg.ses_target
    #     self.get_logger().info("Received new target poses")

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

            # Flatten ids array for easier use
            ids_flat = ids.flatten()

            # ✅ Count markers with ID < 20
            visible_count = np.sum(ids_flat < 20)

            for i, corner in enumerate(corners):

                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, self.marker_size, camera_matrix, dist_coeffs)

                # Draw axis (optional, keep if useful)
                cv2.drawFrameAxes(
                    color_image, camera_matrix, dist_coeffs,
                    rvec[0][0], tvec[0][0], 0.03)

                # Overlay ID text
                position_text = f"ID:{ids_flat[i]}"
                corner_point = tuple(corner[0][0].astype(int))
                cv2.putText(color_image, position_text,
                            corner_point,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)
        else:
            visible_count = 0

                # # ✅ DRAW WHITE INFO BOX
                # box_x, box_y = 20, 20
                # box_w, box_h = 360, 80

        # Publish visible marker count
        msg = Int32()
        msg.data = int(visible_count)
        self.marker_pub.publish(msg)

        # Get image dimensions
        img_h, img_w, _ = color_image.shape

        box_w, box_h = 250, 55
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

        # ✅ Show marker count instead of current angle
        cv2.putText(color_image,
                    f"Visible markers: {visible_count}",
                    (box_x + 10, box_y + 35),
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