#!/usr/bin/env python3

import rospy
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import InteractiveMarkerControl, InteractiveMarker, InteractiveMarkerFeedback
from gazebo_msgs.srv import *
from geometry_msgs.msg import *

def make_marker():
    server = InteractiveMarkerServer("simple_marker")

    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "base_link"
    int_marker.name = "my_marker"
    int_marker.description = "Simple 6-DOF Control"
    int_marker.scale = 0.3

    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        current_pose_srv = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        req = GetLinkStateRequest()
        req.link_name = 'wrist_3_link'
        req.reference_frame = 'base_link'
        resp = current_pose_srv(req)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
    int_marker.pose = resp.link_state.pose

    # control = InteractiveMarkerControl()
    # control.orientation.w = 1
    # control.orientation.x = 1
    # control.name = "move_x"
    # control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    # int_marker.controls.append(control)

    # Create a helper function to add controls for each axis
    def add_6dof_control(marker, axis, mode, name):
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        if axis == 'x':
            control.orientation.x = 1
        elif axis == 'y':
            control.orientation.y = 1
        elif axis == 'z':
            control.orientation.z = 1
        control.name = name
        control.interaction_mode = mode
        marker.controls.append(control)

    # Add rotation controls
    add_6dof_control(int_marker, 'x', InteractiveMarkerControl.ROTATE_AXIS, 'rotate_x')
    add_6dof_control(int_marker, 'y', InteractiveMarkerControl.ROTATE_AXIS, 'rotate_y')
    add_6dof_control(int_marker, 'z', InteractiveMarkerControl.ROTATE_AXIS, 'rotate_z')

    # Add movement controls
    add_6dof_control(int_marker, 'x', InteractiveMarkerControl.MOVE_AXIS, 'move_x')
    add_6dof_control(int_marker, 'y', InteractiveMarkerControl.MOVE_AXIS, 'move_y')
    add_6dof_control(int_marker, 'z', InteractiveMarkerControl.MOVE_AXIS, 'move_z')

    server.insert(int_marker)
    server.applyChanges()

def marker_callback(msg):
    marker_pose = msg.pose
    msg = PoseStamped()
    msg.pose = marker_pose
    header = Header()
    header.seq = 1
    header.frame_id = "base_link"
    msg.header = header
    target_publisher.publish(msg)

if __name__ == "__main__":
    rospy.init_node("interactive_marker_demo")
    make_marker()
    target_publisher = rospy.Publisher('/my_cartesian_motion_controller/target_frame', PoseStamped, queue_size=10)
    rospy.Subscriber("/simple_marker/feedback", InteractiveMarkerFeedback, marker_callback)
    rospy.spin()

    
