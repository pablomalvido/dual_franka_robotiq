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

#include <custom_franka_controllers/cartesian_force_controller.hpp>
#include <custom_franka_controllers/default_robot_behavior_utils.hpp>
#include <custom_franka_controllers/robot_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace custom_franka_controllers {

controller_interface::InterfaceConfiguration
CartesianForceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_velocity_->get_command_interface_names();

  return config;
}

/*
controller_interface::InterfaceConfiguration
CartesianForceController::state_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::NONE};
}
*/

controller_interface::InterfaceConfiguration
CartesianForceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_state_interface_names();
  // add the robot time interface
  //config.names.push_back(arm_id_ + "/robot_time");
  return config;
}

controller_interface::return_type CartesianForceController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& period) {

  //RCLCPP_INFO(get_node()->get_logger(), "time = %.9f", time.seconds());

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
    move_ = false;
  }

  if (move_){
    RCLCPP_INFO(get_node()->get_logger(), "MOVE");
    double dt_ = period.seconds();
    // Get target velocities (latest received values)
    double dx_tgt = dx_target_;
    double dy_tgt = dy_target_;
    double dz_tgt = dz_target_;
    double f_tgt = force_target_;

    // 1. Update the scalar acceleration
    accel_target_ += (f_tgt - force_current_) * 0.0005;

    // 2. Update the scalar speed: v = u + at
    speed_target_ = std::clamp(speed_target_ + accel_target_ * dt_, -max_v_, max_v_);

    // 3. Project scalar speed onto the unit direction vector
    double vx_tgt = speed_target_ * dx_tgt;
    double vy_tgt = speed_target_ * dy_tgt;
    double vz_tgt = speed_target_ * dz_tgt;

    // Exponential smoothing: blend previous command with new target
    vx_cmd_ = (1.0 - smoothing_alpha_) * vx_cmd_ + smoothing_alpha_ * vx_tgt;
    vy_cmd_ = (1.0 - smoothing_alpha_) * vy_cmd_ + smoothing_alpha_ * vy_tgt;
    vz_cmd_ = (1.0 - smoothing_alpha_) * vz_cmd_ + smoothing_alpha_ * vz_tgt;

    // Check if the motion time limit is reached, in this case stop
    elapsed_time_ += dt_;
    if (elapsed_time_ >= time_target_){ //DEFINE TIME_ELAPSED
      move_ = false;
      vx_cmd_ = 0.0;
      vy_cmd_ = 0.0;
      vz_cmd_ = 0.0;
    }
  }
  else{
    RCLCPP_INFO(get_node()->get_logger(), "DO NOT MOVE");
    vx_cmd_ = 0.0;
    vy_cmd_ = 0.0;
    vz_cmd_ = 0.0;
  }

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

CallbackReturn CartesianForceController::on_init() {
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianForceController::on_configure(
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

  target_force_subscr_ = get_node()->create_subscription<custom_msgs::msg::ForceControllerCmd>(
    get_node()->get_name() + std::string("/cmd"), 3,
    std::bind(&CartesianForceController::targetForceCallback, this, std::placeholders::_1));

  current_force_subscr_ = get_node()->create_subscription<geometry_msgs::msg::WrenchStamped>(
    "/nordbo/current", 3,
    std::bind(&CartesianForceController::currentForceCallback, this, std::placeholders::_1));

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianForceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_->assign_loaned_command_interfaces(command_interfaces_);
  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_); //Modified: To read the current pose
  force_target_ = 0.0;
  time_target_ = 0.0;
  move_ = false; //The controller executes pushes for a specific time and then stops moving
  speed_target_ = 0.0;
  accel_target_ = 0.0;
  vx_cmd_ = 0.0;
  vy_cmd_ = 0.0;
  vz_cmd_ = 0.0;
  //elapsed_time_ = rclcpp::Duration(0, 0);
  m_active = true;
  target_update = true;
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianForceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_->release_interfaces();
  franka_cartesian_pose_->release_interfaces(); //
  m_active = false;
  return CallbackReturn::SUCCESS;
}

void CartesianForceController::targetForceCallback(
  const custom_msgs::msg::ForceControllerCmd::SharedPtr target)
{
  if (!this->isActive())
  {
    return;
  }

  RCLCPP_INFO(get_node()->get_logger(), "RECEIVED TARGET SPEED TOPIC");

  if (std::isnan(target->dx) || std::isnan(target->dy) || std::isnan(target->dz) || std::isnan(target->force) || std::isnan(target->time))
  {
    auto & clock = *get_node()->get_clock();
    RCLCPP_WARN_STREAM_THROTTLE(get_node()->get_logger(), clock, 3000,
                                "NaN detected in target speed. Ignoring input.");
    return;
  }

  dx_target_ = static_cast<double>(target->dx);
  dy_target_ = static_cast<double>(target->dy);
  dz_target_ = static_cast<double>(target->dz);
  double d_magnitude = std::sqrt(std::pow(dx_target_,2) + std::pow(dy_target_,2) + std::pow(dz_target_,2));

  if (d_magnitude > 0.0 && std::abs(d_magnitude - 1.0) > 1e-6) {
      dx_target_ /= d_magnitude;
      dy_target_ /= d_magnitude;
      dz_target_ /= d_magnitude;
  }

  force_target_ = std::clamp(static_cast<double>(target->force), 0.0, max_force_);
  time_target_ = static_cast<double>(target->time);
  elapsed_time_ = 0.0;
  target_update = true;
  move_ = true; //The controller executes pushes for a specific time and then stops moving
}

void CartesianForceController::currentForceCallback(
  const geometry_msgs::msg::WrenchStamped::SharedPtr current)
{
  if (!this->isActive())
  {
    return;
  }

  RCLCPP_INFO(get_node()->get_logger(), "RECEIVED CURRENT FORCE");

  if (std::isnan(current->wrench.force.x) || std::isnan(current->wrench.force.y) || std::isnan(current->wrench.force.z))
  {
    auto & clock = *get_node()->get_clock();
    RCLCPP_WARN_STREAM_THROTTLE(get_node()->get_logger(), clock, 3000,
                                "NaN detected in current force. Ignoring input.");
    return;
  }

  force_current_ = std::sqrt(std::pow(current->wrench.force.x, 2) + std::pow(current->wrench.force.y, 2) + std::pow(current->wrench.force.z, 2));
  RCLCPP_INFO(get_node()->get_logger(), "RECEIVED CURRENT FORCE: %.9f", force_current_);
}
 
}  // namespace custom_franka_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(custom_franka_controllers::CartesianForceController,
                       controller_interface::ControllerInterface)
