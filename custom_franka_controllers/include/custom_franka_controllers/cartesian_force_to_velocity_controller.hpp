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
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <franka_semantic_components/franka_cartesian_velocity_interface.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace custom_franka_controllers {

/**
 * The cartesian velocity example controller
 */
class CartesianForceToVelocityController : public controller_interface::ControllerInterface {
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

  const double k_time_max_{4.0};
  const double k_v_max_{0.05};
  const double k_angle_{M_PI / 4.0};
  const bool k_elbow_activated_{false};

  // State variables
  rclcpp::Time last_sensor_time_;
  const rclcpp::Duration t_sensor_max{0, 5000000}; //0.05s {seconds, nanoseconds}

  // Target and current forces values (from topic subscriber)
  double fx_target_ = 0.0;
  double fy_target_ = 0.0;
  double fz_target_ = 0.0;
  std::atomic<double> fx_current_{0.0};
  std::atomic<double> fy_current_{0.0};
  std::atomic<double> fz_current_{0.0};

  // Maximum allowed values (limits)
  const double max_vx_ = 0.05;
  const double max_vy_ = 0.05;
  const double max_vz_ = 0.05;
  const double k_fv = 0.005;

  // Smoothed current command velocities
  double vx_cmd_ = 0.0;
  double vy_cmd_ = 0.0;
  double vz_cmd_ = 0.0;

  // Smoothing factor (0.0 = no change, 1.0 = immediate change)
  double smoothing_alpha_ = 0.02;

  // ROS subscriber
  void targetForceCallback(const geometry_msgs::msg::Vector3::SharedPtr target);
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr target_force_subscr_;

  void currentForceCallback(const geometry_msgs::msg::WrenchStamped::SharedPtr w_current);
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr current_force_subscr_;

  bool isActive() const { return m_active; };
  bool m_active = {false};

};

}  // namespace custom_franka_controllers
