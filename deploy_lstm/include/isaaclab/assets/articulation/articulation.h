// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

namespace isaaclab
{

class MotionLoader;

struct ArticulationData
{
    Eigen::Vector3f GRAVITY_VEC_W = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f FORWARD_VEC_B = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    std::vector<float> joint_stiffness; // sdk order
    std::vector<float> joint_damping; // sdk order

    // Joint positions of all joints.
    Eigen::VectorXf joint_pos;
    
    // Default joint positions of all joints.
    Eigen::VectorXf default_joint_pos;

    // Joint velocities of all joints.
    Eigen::VectorXf joint_vel;

    // Root angular velocity in base world frame.
    Eigen::Vector3f root_ang_vel_b;

    // Projection of the gravity direction on base frame.
    Eigen::Vector3f projected_gravity_b;

    Eigen::Quaternionf root_quat_w;

    std::vector<float> joint_ids_map;

    unitree::common::UnitreeJoystick* joystick = nullptr;

    // ── FK-derived body-frame foot positions (for stair_latent obs) ──
    Eigen::Vector3f left_toe_pos_body    = Eigen::Vector3f::Zero();
    Eigen::Vector3f right_toe_pos_body   = Eigen::Vector3f::Zero();
    Eigen::Vector3f left_heel_pos_body   = Eigen::Vector3f::Zero();
    Eigen::Vector3f right_heel_pos_body  = Eigen::Vector3f::Zero();

    // ── Previous-frame caches for delta features ──
    Eigen::Vector3f prev_left_toe_pos_body  = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_right_toe_pos_body = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_base_ang_vel       = Eigen::Vector3f::Zero();
    Eigen::VectorXf prev_leg_joint_vel;      // 12-DOF leg joint vel (resized on init)

    // ── Previous toe velocity caches (for toe_vel_delta features) ──
    Eigen::Vector3f prev_left_toe_vel   = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_right_toe_vel  = Eigen::Vector3f::Zero();
    bool            prev_toe_vel_valid  = false;

    // ── Previous action cache (full 29-DOF, for last_action obs term) ──
    std::vector<float> prev_action;          // resized to joint_ids_map.size() on init

    // ── stair_latent cache validity flag ──
    bool stair_latent_cache_valid = false;
};

class Articulation
{
public:
    Articulation(){}

    virtual void update(){};

    ArticulationData data;
};

};