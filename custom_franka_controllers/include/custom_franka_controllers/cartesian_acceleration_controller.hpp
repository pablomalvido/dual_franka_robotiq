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

#pragma once

#include <string>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <franka_semantic_components/franka_cartesian_velocity_interface.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp> //

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace custom_franka_controllers {

/**
 * The cartesian velocity example controller
 */
class CartesianAccelerationController : public controller_interface::ControllerInterface {
 public:
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
  std::unique_ptr<franka_semantic_components::FrankaCartesianVelocityInterface>
      franka_cartesian_velocity_;

  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_; //
  Eigen::Quaterniond orientation_; //
  Eigen::Vector3d position_; //

  const double k_time_max_{4.0};
  const double k_v_max_{0.05};
  const double k_angle_{M_PI / 4.0};
  const bool k_elbow_activated_{false};

  // State variables
  rclcpp::Duration elapsed_time_ = rclcpp::Duration(0, 0);

  // Target velocity values (from topic subscriber)
  std::atomic<double> vx_target_{0.0};
  std::atomic<double> vy_target_{0.0};
  std::atomic<double> vz_target_{0.0};
  double vx_target_test_{0.0};

  // Maximum allowed values (limits)
  const double max_vx_ = 0.05;
  const double max_vy_ = 0.05;
  const double max_vz_ = 0.05;

  // Smoothed current command velocities
  double vx_cmd_ = 0.0;
  double vy_cmd_ = 0.0;
  double vz_cmd_ = 0.0;

  // Smoothing factor (0.0 = no change, 1.0 = immediate change)
  double smoothing_alpha_ = 0.03;

  // ROS subscriber
  void targetSpeedCallback(const geometry_msgs::msg::Vector3::SharedPtr target);
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr target_speed_subscr_;

  bool isActive() const { return m_active; };
  bool m_active = {false};

  //Modified lines from pose controller
  std::string arm_id_;
  Eigen::Vector3d start_position_;
  bool target_update = {false};
  const double max_x_displacement_ = 0.2;
  const double max_x_pos_ = 0.65;
};

}  // namespace custom_franka_controllers
