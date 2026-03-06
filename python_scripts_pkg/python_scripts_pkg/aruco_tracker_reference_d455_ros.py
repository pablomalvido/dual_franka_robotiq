import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray, Pose
from custom_msgs.msg import SesStarter, MarkerPoseArray, MarkerPose
import cv2
import numpy as np
import pyrealsense2 as rs

class ArucoPublisher(Node):

    def __init__(self):
        super().__init__('aruco_publisher')
        # Publisher of aruco poses
        self.marker_pub = self.create_publisher(MarkerPoseArray, 'ses/aruco_poses', 10)
        
        # Subscriber to 2D poses
        self.subscription = self.create_subscription(
            SesStarter,
            "ses/activate",
            self.ref_poses_callback,
            10)

        self.latest_target_poses = None
        self.default_ref_pose() #Generate a default pose

        # Distance threshold (meters)
        self.distance_threshold = 0.05 #m
        
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

    def ref_poses_callback(self, msg):
        print(msg)
        self.latest_target_poses = msg.ses_target
        self.get_logger().info("Received new target poses")
    
    def default_ref_pose(self):
        # Create default PoseArray
        self.latest_target_poses = PoseArray()
        self.latest_target_poses.header.frame_id = "camera_link"
        self.latest_target_poses.header.stamp = self.get_clock().now().to_msg()

        # default_positions = [
        #     (0.15, -0.23, 0.0),
        #     (0.1, -0.1, 0.0),
        #     (0.05, 0.03, 0.0)
        # ]
        # default_positions = [
        #     (0.2, -0.16, 0.0),
        #     (0.05, -0.1, 0.0),
        #     (0.01, 0.03, 0.0)
        # ]
        default_positions = [ #Chain
            (-0.08, -0.23, 0.0),
            (0.01, -0.16, 0.0),
            (0.09, 0.01, 0.0)
        ]

        for pos in default_positions:
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = pos[2]

            # Default orientation (identity quaternion)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0

            self.latest_target_poses.poses.append(pose)
        

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

                # Draw axis
                cv2.drawFrameAxes(
                    color_image, camera_matrix, dist_coeffs,
                    rvec[0][0], tvec[0][0], 0.03)

                # Overlay XYZ text
                position_text = f"ID:{ids[i][0]} X:{tvec[0][0][0]:.2f} Y:{tvec[0][0][1]:.2f} Z:{tvec[0][0][2]:.2f}"
                corner_point = tuple(corner[0][0].astype(int))
                cv2.putText(color_image, position_text,
                            corner_point,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

            #Publish all markers
            marker_array_msg = MarkerPoseArray()
            marker_array_msg.header.stamp = self.get_clock().now().to_msg()
            marker_array_msg.header.frame_id = "camera_frame"
            for marker in marker_positions:
                marker_id, x, y, z = marker
                marker_msg = MarkerPose()
                marker_msg.id = int(marker_id)
                marker_msg.pose.position.x = x
                marker_msg.pose.position.y = y
                marker_msg.pose.position.z = z
                marker_msg.pose.orientation.x = 0.0
                marker_msg.pose.orientation.y = 0.0
                marker_msg.pose.orientation.z = 0.0
                marker_msg.pose.orientation.w = 1.0
                marker_array_msg.markers.append(marker_msg)
            self.marker_pub.publish(marker_array_msg)

            # Compare with latest received poses
            if self.latest_target_poses is not None:

                num_markers = min(len(marker_positions),
                                len(self.latest_target_poses.poses),
                                3)  # n = 3
                marker_positions.sort(key=lambda m: m[0])

                for i in range(num_markers):

                    target_pose = self.latest_target_poses.poses[i]

                    target_x = target_pose.position.x
                    target_y = target_pose.position.y

                    marker_x = marker_positions[i][1]
                    marker_y = marker_positions[i][2]

                    # print("##")
                    # print(target_y)
                    # print(marker_y)

                    # Euclidean distance (2D)
                    distance = np.sqrt(
                        (target_x - marker_x) ** 2 +
                        (target_y - marker_y) ** 2
                    )

                    # Choose color
                    if distance < self.distance_threshold:
                        color = (255, 0, 0)  # Blue
                    else:
                        color = (0, 0, 255)  # Red

                    # Project 3D target point to image
                    point_3d = np.array([[target_x, target_y, marker_positions[i][3]]],
                                        dtype=np.float32)

                    imgpts, _ = cv2.projectPoints(
                        point_3d,
                        np.zeros((3, 1)),
                        np.zeros((3, 1)),
                        camera_matrix,
                        dist_coeffs)

                    center = tuple(imgpts[0][0].astype(int))

                    cv2.circle(color_image, center, 10, color, -1)

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