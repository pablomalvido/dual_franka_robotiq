// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Dense>
#include <string>

#include <controller_interface/controller_interface.hpp>
#include <custom_franka_controllers/robot_utils_impedance.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include "franka_semantic_components/franka_cartesian_pose_interface.hpp"
#include "franka_semantic_components/franka_robot_model.hpp"

#include <tf2/LinearMath/Vector3.h>
#include <urdf/model.h>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl_parser/kdl_parser.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace custom_franka_controllers {

/**
 * joint impedance controller to move the robot to a desired pose.
 */
class JointImpedanceIKController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  /**
   * @brief updates the joint states from the state interfaces
   *
   */
  void update_joint_states_();

  /**
   * @brief computes the torque commands based on impedance control law with compensated coriolis
   * terms
   *
   * @return Eigen::Vector7d torque for each joint of the robot
   */
  Vector7d compute_torque_command_(const Vector7d& joint_positions_desired,
                                   const Vector7d& joint_positions_current,
                                   const Vector7d& joint_velocities_current);

  /**
   * @brief assigns the Kp, Kd and arm_id parameters
   *
   * @return true when parameters are present, false when parameters are not available
   */
  bool assign_parameters_();

  tf2::Vector3 transform_velocity_to_world_frame_(
      const geometry_msgs::msg::Twist::SharedPtr& msg) const;

  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;

  Eigen::Quaterniond orientation_;
  Eigen::Vector3d position_;

  const bool k_elbow_activated_{false};

  std::string arm_id_;
  std::string namespace_prefix_;
  urdf::Model model_;
  KDL::Tree tree_;
  KDL::Chain chain_;
  unsigned int nj_;
  KDL::JntArray q_min_, q_max_, q_init_, q_result_;
  void solve_ik_(const Eigen::Vector3d& new_position, const Eigen::Quaterniond& new_orientation);
  bool is_gripper_loaded_ = true;
  std::vector<double> arm_mounting_orientation_;

  std::string robot_description_;
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};

  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  const int num_joints_{7};

  std::vector<double> joint_positions_desired_;
  std::vector<double> joint_positions_current_{0, 0, 0, 0, 0, 0, 0};
  std::vector<double> joint_velocities_current_{0, 0, 0, 0, 0, 0, 0};
  std::vector<double> joint_efforts_current_{0, 0, 0, 0, 0, 0, 0};

  // Spacemouse
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr spacemouse_sub_;
  void spacemouse_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr rviz_teleop_sub_;
  void targetPosCallback(const geometry_msgs::msg::Vector3::SharedPtr msg);
  bool rviz_target = {false};
  Eigen::Vector3d desired_linear_position_update_{0.0, 0.0, 0.0};
  Eigen::Vector3d desired_angular_position_update_{0.0, 0.0, 0.0};
  Eigen::Quaterniond desired_angular_position_update_quaternion_{1.0, 0.0, 0.0, 0.0};
  Eigen::Vector3d desired_linear_position_{0.0, 0.0, 0.0};
};
}  // namespace custom_franka_controllers