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

#include <custom_franka_controllers/cartesian_force_to_velocity_controller.hpp>
#include <custom_franka_controllers/default_robot_behavior_utils.hpp>
#include <custom_franka_controllers/robot_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace custom_franka_controllers {

controller_interface::InterfaceConfiguration
CartesianForceToVelocityController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_velocity_->get_command_interface_names();

  return config;
}

controller_interface::InterfaceConfiguration
CartesianForceToVelocityController::state_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::NONE};
}

controller_interface::return_type CartesianForceToVelocityController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& /*period*/) {

  // if no force sensor readings in t_sensor_max, return an error
  /*rclcpp::Duration sensor_elapsed_time_ = get_node()->now() - last_sensor_time_;
  if (sensor_elapsed_time_ > t_sensor_max) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "No values received from the force sensor in the expected time");
    return controller_interface::return_type::ERROR;
  }*/    

  // Compute target velocities
  double vx_target_ = k_fv*(fx_current_ - fx_target_); //Opposite direction
  double vy_target_ = k_fv*(fy_current_ - fy_target_);
  double vz_target_ = k_fv*(fz_current_ - fz_target_);

  double vx_tgt = std::clamp(vx_target_, -max_vx_, max_vx_);
  double vy_tgt = std::clamp(vy_target_, -max_vy_, max_vy_);
  double vz_tgt = std::clamp(vz_target_, -max_vz_, max_vz_);

  // Exponential smoothing: blend previous command with new target
  vx_cmd_ = (1.0 - smoothing_alpha_) * vx_cmd_ + smoothing_alpha_ * vx_tgt;
  vy_cmd_ = (1.0 - smoothing_alpha_) * vy_cmd_ + smoothing_alpha_ * vy_tgt;
  vz_cmd_ = (1.0 - smoothing_alpha_) * vz_cmd_ + smoothing_alpha_ * vz_tgt;

  // Build and send the velocity command
  Eigen::Vector3d cartesian_linear_velocity(vx_cmd_, vy_cmd_, vz_cmd_);
  Eigen::Vector3d cartesian_angular_velocity(0.0, 0.0, 0.0);

  RCLCPP_INFO(get_node()->get_logger(), "CMD VELOCITIES: %.3f, %.3f, %.3f", vx_cmd_, vy_cmd_, vz_cmd_);

  if (franka_cartesian_velocity_->setCommand(cartesian_linear_velocity,
                                             cartesian_angular_velocity)) {
    return controller_interface::return_type::OK;
  } else {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Set command failed. Did you activate the elbow command interface?");
    return controller_interface::return_type::ERROR;
  }
}

CallbackReturn CartesianForceToVelocityController::on_init() {
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianForceToVelocityController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_ =
      std::make_unique<franka_semantic_components::FrankaCartesianVelocityInterface>(
          franka_semantic_components::FrankaCartesianVelocityInterface(k_elbow_activated_));

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

  target_force_subscr_ = get_node()->create_subscription<geometry_msgs::msg::Vector3>(
    get_node()->get_name() + std::string("/target_force"), 3,
    std::bind(&CartesianForceToVelocityController::targetForceCallback, this, std::placeholders::_1));

  current_force_subscr_ = get_node()->create_subscription<geometry_msgs::msg::WrenchStamped>(
    get_node()->get_name() + std::string("/current_force"), 3,
    std::bind(&CartesianForceToVelocityController::currentForceCallback, this, std::placeholders::_1));

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianForceToVelocityController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_->assign_loaned_command_interfaces(command_interfaces_);
  fx_target_ = fx_current_;
  fy_target_ = fx_current_;
  fz_target_ = fx_current_;
  m_active = true;
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianForceToVelocityController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_->release_interfaces();
  m_active = false;
  return CallbackReturn::SUCCESS;
}


void CartesianForceToVelocityController::targetForceCallback(
  const geometry_msgs::msg::Vector3::SharedPtr target)
{
  if (!this->isActive())
  {
    return;
  }

  RCLCPP_INFO(get_node()->get_logger(), "RECEIVED TARGET FORCE TOPIC");

  if (std::isnan(target->x) || std::isnan(target->y) || std::isnan(target->z))
  {
    auto & clock = *get_node()->get_clock();
    RCLCPP_WARN_STREAM_THROTTLE(get_node()->get_logger(), clock, 3000,
                                "NaN detected in target force. Ignoring input.");
    return;
  }

  fx_target_ = static_cast<double>(target->x);
  fy_target_ = static_cast<double>(target->y);
  fz_target_ = static_cast<double>(target->z);
}


void CartesianForceToVelocityController::currentForceCallback(
  const geometry_msgs::msg::WrenchStamped::SharedPtr w_current)
{
  if (!this->isActive())
  {
    return;
  }

  RCLCPP_INFO(get_node()->get_logger(), "RECEIVED CURRENT FORCE TOPIC");

  if (std::isnan(w_current->wrench.force.x) || std::isnan(w_current->wrench.force.y) || std::isnan(w_current->wrench.force.z))
  {
    auto & clock = *get_node()->get_clock();
    RCLCPP_WARN_STREAM_THROTTLE(get_node()->get_logger(), clock, 3000,
                                "NaN detected in current force. Ignoring input.");
    return;
  }

  fx_current_ = static_cast<double>(w_current->wrench.force.x);
  fy_current_ = static_cast<double>(w_current->wrench.force.y);
  fz_current_ = static_cast<double>(w_current->wrench.force.z);

  //last_sensor_time_ = rclcpp::Time(w_current->header.stamp, rclcpp::Clock(RCL_SYSTEM_TIME)); 
  //last_sensor_time_ = rclcpp::Time(w_current->header.stamp, RCL_SYSTEM_TIME);
  last_sensor_time_ = get_node()->now();
}

}  // namespace custom_franka_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(custom_franka_controllers::CartesianForceToVelocityController,
                       controller_interface::ControllerInterface)
