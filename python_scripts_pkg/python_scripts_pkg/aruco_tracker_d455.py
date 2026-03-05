import pyrealsense2 as rs
import numpy as np
import cv2
import rclpy

# ----------------------------
# ROS init
# ----------------------------
features_pub

import pyrealsense2 as rs
import numpy as np
import cv2

# ----------------------------
# Marker settings
# ----------------------------
marker_length = 0.04  # meters (4 cm marker)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ----------------------------
# RealSense D455 pipeline
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()

# D455 color stream (recommended 1280x720 or 1920x1080)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Get intrinsics from D455 color sensor
color_stream = profile.get_stream(rs.stream.color)
video_profile = color_stream.as_video_stream_profile()
intrinsics = video_profile.get_intrinsics()

camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array(intrinsics.coeffs)

print("D455 Camera matrix:\n", camera_matrix)
print("D455 Distortion coeffs:\n", dist_coeffs)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                marker_length,
                camera_matrix,
                dist_coeffs
            )

            cv2.aruco.drawDetectedMarkers(frame, corners)

            for i in range(len(ids)):
                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    marker_length * 0.5
                )

                print(f"Marker ID: {ids[i][0]}")
                print("Translation (m):", tvecs[i].flatten())
                print("Rotation (rvec):", rvecs[i].flatten())
                print("----")

        cv2.imshow("RealSense D455 ArUco Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()