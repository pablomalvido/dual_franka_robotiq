// Copyright (c) 2023 Franka Robotics GmbH
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

#include <custom_franka_controllers/cartesian_velocity_example_controller.hpp>
#include <custom_franka_controllers/default_robot_behavior_utils.hpp>
#include <custom_franka_controllers/robot_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace custom_franka_controllers {

controller_interface::InterfaceConfiguration
CartesianVelocityExampleController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_velocity_->get_command_interface_names();

  return config;
}

/*
controller_interface::InterfaceConfiguration
CartesianVelocityExampleController::state_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::NONE};
}
*/

controller_interface::InterfaceConfiguration
CartesianVelocityExampleController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_state_interface_names();
  // add the robot time interface
  //config.names.push_back(arm_id_ + "/robot_time");
  return config;
}

controller_interface::return_type CartesianVelocityExampleController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& /*period*/) {

  RCLCPP_INFO(get_node()->get_logger(), "time = %.9f", time.seconds());

  std::tie(orientation_, position_) =
        franka_cartesian_pose_->getCurrentOrientationAndTranslation();
  
  //RCLCPP_INFO(get_node()->get_logger(), "POSE: %.3f, %.3f, %.3f", position_(0), position_(1), position_(2));

  if (target_update){
    RCLCPP_INFO(get_node()->get_logger(), "UPDATE POSE");
    start_position_ = position_;
    target_update = false;
  }

  if ((std::abs(start_position_(0) - position_(0)) > max_x_displacement_) or position_(0) > max_x_pos_){
    RCLCPP_INFO(get_node()->get_logger(), "THRESHOLD EXCEEDED");
    vx_target_ = 0.0;
    vy_target_ = 0.0;
    vz_target_ = 0.0;
  }

  // Get target velocities (latest received values)
  double vx_tgt = vx_target_;
  double vy_tgt = vy_target_;
  double vz_tgt = vz_target_;

  // Exponential smoothing: blend previous command with new target
  vx_cmd_ = (1.0 - smoothing_alpha_) * vx_cmd_ + smoothing_alpha_ * vx_tgt;
  vy_cmd_ = (1.0 - smoothing_alpha_) * vy_cmd_ + smoothing_alpha_ * vy_tgt;
  vz_cmd_ = (1.0 - smoothing_alpha_) * vz_cmd_ + smoothing_alpha_ * vz_tgt;

  // Build and send the velocity command
  Eigen::Vector3d cartesian_linear_velocity(vx_cmd_, vy_cmd_, vz_cmd_);
  Eigen::Vector3d cartesian_angular_velocity(0.0, 0.0, 0.0);

  //RCLCPP_INFO(get_node()->get_logger(), "CMD VELOCITIES: %.3f, %.3f, %.3f", vx_cmd_, vy_cmd_, vz_cmd_);

  if (franka_cartesian_velocity_->setCommand(cartesian_linear_velocity,
                                             cartesian_angular_velocity)) {
    return controller_interface::return_type::OK;
  } else {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Set command failed. Did you activate the elbow command interface?");
    return controller_interface::return_type::ERROR;
  }
}

CallbackReturn CartesianVelocityExampleController::on_init() {
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianVelocityExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_ =
      std::make_unique<franka_semantic_components::FrankaCartesianVelocityInterface>(
          franka_semantic_components::FrankaCartesianVelocityInterface(k_elbow_activated_));
  
  franka_cartesian_pose_ =
      std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
          franka_semantic_components::FrankaCartesianPoseInterface(k_elbow_activated_));

  auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
      "service_server/set_full_collision_behavior");
  auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();

  auto future_result = client->async_send_request(request);
  future_result.wait_for(robot_utils::time_out);

  auto success = future_result.get();
  if (!success) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
    return CallbackReturn::ERROR;
  } else {
    RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");
  }

  target_speed_subscr_ = get_node()->create_subscription<geometry_msgs::msg::Vector3>(
    get_node()->get_name() + std::string("/target_speed"), 3,
    std::bind(&CartesianVelocityExampleController::targetSpeedCallback, this, std::placeholders::_1));

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianVelocityExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_->assign_loaned_command_interfaces(command_interfaces_);
  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_); //Modified: To read the current pose
  vx_target_ = 0.0;
  vy_target_ = 0.0;
  vz_target_ = 0.0;
  vx_cmd_ = 0.0;
  vy_cmd_ = 0.0;
  vz_cmd_ = 0.0;
  //elapsed_time_ = rclcpp::Duration(0, 0);
  m_active = true;
  target_update = true;
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianVelocityExampleController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_->release_interfaces();
  franka_cartesian_pose_->release_interfaces(); //
  m_active = false;
  return CallbackReturn::SUCCESS;
}

void CartesianVelocityExampleController::targetSpeedCallback(
  const geometry_msgs::msg::Vector3::SharedPtr target)
{
  if (!this->isActive())
  {
    return;
  }

  RCLCPP_INFO(get_node()->get_logger(), "RECEIVED TARGET SPEED TOPIC");

  if (std::isnan(target->x) || std::isnan(target->y) || std::isnan(target->z))
  {
    auto & clock = *get_node()->get_clock();
    RCLCPP_WARN_STREAM_THROTTLE(get_node()->get_logger(), clock, 3000,
                                "NaN detected in target speed. Ignoring input.");
    return;
  }

  vx_target_ = std::clamp(static_cast<double>(target->x), -max_vx_, max_vx_);
  vy_target_ = std::clamp(static_cast<double>(target->y), -max_vy_, max_vy_);
  vz_target_ = std::clamp(static_cast<double>(target->z), -max_vz_, max_vz_);
  target_update = true;
}

}  // namespace custom_franka_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(custom_franka_controllers::CartesianVelocityExampleController,
                       controller_interface::ControllerInterface)
