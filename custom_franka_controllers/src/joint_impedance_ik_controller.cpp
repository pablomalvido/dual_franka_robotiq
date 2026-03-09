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

#include <custom_franka_controllers/default_robot_behavior_utils.hpp>
#include <custom_franka_controllers/joint_impedance_ik_controller.hpp>

#include <chrono>
#include <string>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace std::chrono_literals;
using Vector7d = Eigen::Matrix<double, 7, 1>;

namespace custom_franka_controllers {

controller_interface::InterfaceConfiguration
JointImpedanceIKController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceIKController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_state_interface_names();
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/position");
  }
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/velocity");
  }
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }

  return config;
}

void JointImpedanceIKController::update_joint_states_() {
  for (auto i = 0; i < num_joints_; ++i) {
    const auto& position_interface = state_interfaces_.at(16 + i);
    const auto& velocity_interface = state_interfaces_.at(23 + i);
    const auto& effort_interface = state_interfaces_.at(30 + i);
    joint_positions_current_[i] = position_interface.get_value();
    q_init_(i) = joint_positions_current_[i];
    joint_velocities_current_[i] = velocity_interface.get_value();
    joint_efforts_current_[i] = effort_interface.get_value();
  }
}

Vector7d JointImpedanceIKController::compute_torque_command_(
    const Vector7d& joint_positions_desired,
    const Vector7d& joint_positions_current,
    const Vector7d& joint_velocities_current) {
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  Vector7d coriolis(coriolis_array.data());
  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * joint_velocities_current;
  Vector7d q_error = joint_positions_desired - joint_positions_current;
  Vector7d tau_d_calculated =
      k_gains_.cwiseProduct(q_error) - d_gains_.cwiseProduct(dq_filtered_) + coriolis;

  return tau_d_calculated;
}

controller_interface::return_type JointImpedanceIKController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {
  update_joint_states_();
  std::tie(orientation_, position_) = franka_cartesian_pose_->getCurrentOrientationAndTranslation();

  auto new_position = position_;
  auto new_orientation = orientation_;
  if (rviz_target){
    new_position = desired_linear_position_;
  }
  else{
    new_position = position_ + desired_linear_position_update_;
    new_orientation = orientation_ * desired_angular_position_update_quaternion_;
  }

  solve_ik_(new_position, new_orientation);

  if (joint_positions_desired_.empty()) {
    return controller_interface::return_type::OK;
  }

  Vector7d joint_positions_desired_eigen(joint_positions_desired_.data());
  Vector7d joint_positions_current_eigen(joint_positions_current_.data());
  Vector7d joint_velocities_current_eigen(joint_velocities_current_.data());

  auto tau_d_calculated = compute_torque_command_(
      joint_positions_desired_eigen, joint_positions_current_eigen, joint_velocities_current_eigen);

  for (int i = 0; i < num_joints_; i++) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }

  return controller_interface::return_type::OK;
}

CallbackReturn JointImpedanceIKController::on_init() {
  franka_cartesian_pose_ =
      std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
          franka_semantic_components::FrankaCartesianPoseInterface(k_elbow_activated_));

  return CallbackReturn::SUCCESS;
}

bool JointImpedanceIKController::assign_parameters_() {
  arm_id_ = get_node()->get_parameter("arm_id").as_string() == "fr3";
  is_gripper_loaded_ = get_node()->get_parameter("load_gripper").as_string() == "false";
  arm_mounting_orientation_ =
      get_node()->get_parameter("arm_mounting_orientation").as_double_array();

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  if (k_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
    return false;
  }
  if (k_gains.size() != static_cast<uint>(num_joints_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size %d but is of size %ld",
                 num_joints_, k_gains.size());
    return false;
  }
  if (d_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
    return false;
  }
  if (d_gains.size() != static_cast<uint>(num_joints_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size %d but is of size %ld",
                 num_joints_, d_gains.size());
    return false;
  }
  for (int i = 0; i < num_joints_; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
  }
  return true;
}

CallbackReturn JointImpedanceIKController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (!assign_parameters_()) {
    return CallbackReturn::FAILURE;
  }

  namespace_prefix_ = get_node()->get_namespace();

  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    // Remove leading slash and add trailing underscore
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  /*franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name,
                                                   arm_id_ + "/" + k_robot_state_interface_name));*/

  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel("fr3/" + k_robot_model_interface_name,
                                                   "fr3/" + k_robot_state_interface_name));

  auto collision_client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
      "service_server/set_full_collision_behavior");

  auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();
  auto future_result = collision_client->async_send_request(request);

  auto success = future_result.get();

  if (!success->success) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
    return CallbackReturn::ERROR;
  } else {
    RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");
  }

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  //arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
  arm_id_ = "fr3";

  spacemouse_sub_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
      "franka_controller/target_cartesian_velocity_percent", 10,
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) { this->spacemouse_callback(msg); });

  rviz_teleop_sub_ = get_node()->create_subscription<geometry_msgs::msg::Vector3>(
      "cartesian_velocity_controller_ses/target_pos", 3,
      std::bind(&JointImpedanceIKController::targetPosCallback, this, std::placeholders::_1));

  RCLCPP_INFO(get_node()->get_logger(),
              "Subscribed to franka_controller/target_cartesian_velocity_percent.");

  if (!model_.initString(robot_description_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to parse processed URDF");
    return CallbackReturn::FAILURE;
  }

  if (!kdl_parser::treeFromUrdfModel(model_, tree_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to convert URDF to KDL tree.");
    return CallbackReturn::FAILURE;
  }

  //std::string tcp_name = namespace_prefix_ + "fr3_hand_tcp";
  std::string base_name = namespace_prefix_ + "fr3_link0";
  std::string tcp_name = namespace_prefix_ + "nordbo_ft_sensor_link";
  if (!tree_.getChain(base_name, tcp_name, chain_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to extract KDL chain.");
    return CallbackReturn::FAILURE;
  }

  nj_ = chain_.getNrOfJoints();
  q_min_ = KDL::JntArray(nj_);
  q_max_ = KDL::JntArray(nj_);
  q_init_ = KDL::JntArray(nj_);
  q_result_ = KDL::JntArray(nj_);

  // Extract joint limits from URDF
  unsigned int j = 0;
  for (const auto& segment : chain_.segments) {
    const KDL::Joint& kdl_joint = segment.getJoint();
    if (kdl_joint.getType() == KDL::Joint::None) {
      continue;
    }

    const std::string& joint_name = kdl_joint.getName();
    auto joint = model_.getJoint(joint_name);
    if (!joint || !joint->limits) {
      RCLCPP_FATAL(get_node()->get_logger(), "No limits found for joint: %s", joint_name.c_str());
      return CallbackReturn::FAILURE;
    }

    q_min_(j) = joint->limits->lower;
    q_max_(j) = joint->limits->upper;
    q_init_(j) = (q_max_(j) + q_min_(j)) / 2;
    ++j;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceIKController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  dq_filtered_.setZero();
  joint_positions_desired_.reserve(num_joints_);
  joint_positions_current_.reserve(num_joints_);
  joint_velocities_current_.reserve(num_joints_);
  joint_efforts_current_.reserve(num_joints_);

  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointImpedanceIKController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_pose_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

void JointImpedanceIKController::spacemouse_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg) {
  // The values for max_linear_pos_update and max_angular_pos_update are empirically determined.
  // These values represent a tradeoff between precision in teleop control and speed.
  // Lowering these values will make the robot move slower and more precise, while increasing them
  // will make it move faster but less precise.
  const double max_linear_pos_update = 0.007;
  const double max_angular_pos_update = 0.03;

  rviz_target = false;
  tf2::Vector3 v_linear_world = transform_velocity_to_world_frame_(msg);

  desired_angular_position_update_ =
      max_angular_pos_update * Eigen::Vector3d(msg->angular.x, msg->angular.y, msg->angular.z);
  desired_linear_position_update_ =
      max_linear_pos_update *
      Eigen::Vector3d(v_linear_world.x(), v_linear_world.y(), v_linear_world.z());

  Eigen::AngleAxisd rollAngle(desired_angular_position_update_.x(), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(desired_angular_position_update_.y(), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(desired_angular_position_update_.z(), Eigen::Vector3d::UnitZ());
  desired_angular_position_update_quaternion_ = yawAngle * pitchAngle * rollAngle;
}

void JointImpedanceIKController::targetPosCallback(
    const geometry_msgs::msg::Vector3::SharedPtr msg) {
  const double max_linear_pos_update_rviz = 0.007;
  rviz_target = true;

  Eigen::Vector3d msg_vec(msg->x, msg->y, msg->z);

  Eigen::Vector3d delta = msg_vec - position_;

  double dist = delta.norm();

  if (dist > max_linear_pos_update_rviz && dist > 0.0) {
    delta *= (max_linear_pos_update_rviz / dist);
  }

  desired_linear_position_ = position_ + delta;
}

tf2::Vector3 JointImpedanceIKController::transform_velocity_to_world_frame_(
    const geometry_msgs::msg::Twist::SharedPtr& msg) const {
  tf2::Quaternion q;
  q.setRPY(arm_mounting_orientation_[0], arm_mounting_orientation_[1],
           arm_mounting_orientation_[2]);

  // Create rotation matrix from quaternion
  tf2::Matrix3x3 rotation_matrix(q);

  // Invert the transformation from robot-frame to world-frame
  rotation_matrix = rotation_matrix.transpose();

  tf2::Vector3 v_linear_robot(msg->linear.x, msg->linear.y, msg->linear.z);

  tf2::Vector3 v_linear_world = rotation_matrix * v_linear_robot;

  return v_linear_world;
}

void JointImpedanceIKController::solve_ik_(const Eigen::Vector3d& new_position,
                                           const Eigen::Quaterniond& new_orientation) {
  KDL::ChainFkSolverPos_recursive fk_solver(chain_);
  KDL::ChainIkSolverVel_pinv vel_solver(chain_);
  KDL::ChainIkSolverPos_NR_JL ik_solver(chain_, q_min_, q_max_, fk_solver, vel_solver, 100, 1e-6);

  KDL::Rotation kdl_rot = KDL::Rotation::Quaternion(new_orientation.x(), new_orientation.y(),
                                                    new_orientation.z(), new_orientation.w());
  KDL::Vector kdl_pos(new_position.x(), new_position.y(), new_position.z());
  KDL::Frame desired_pose(kdl_rot, kdl_pos);

  int status = ik_solver.CartToJnt(q_init_, desired_pose, q_result_);
  if (status < 0) {
    RCLCPP_FATAL(get_node()->get_logger(), "IK Failed with error code: %d", status);
    throw std::runtime_error("IK Failed");
  }

  std::vector<double> joint_vector(q_result_.data.data(), q_result_.data.data() + q_result_.rows());

  joint_positions_desired_ = joint_vector;
}

}  // namespace custom_franka_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(custom_franka_controllers::JointImpedanceIKController,
                       controller_interface::ControllerInterface)