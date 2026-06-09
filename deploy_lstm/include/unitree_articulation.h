// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/utils/fk_utils.h"

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);

        // base_angular_velocity
        for (int i(0); i < 3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;

        // joint positions and velocities
        for (int i(0); i < data.joint_ids_map.size(); i++) {
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }

        // ── Compute body-frame foot positions via FK ──
        // We assume the joint order matches the G1 29-DOF convention:
        //   0..5:   left  leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        //   6..11:  right leg
        //   12..14: waist (yaw, roll, pitch)
        //   15..21: left  arm
        //   22..28: right arm
        if (data.joint_ids_map.size() >= 12) {
            float left_angles[6];
            float right_angles[6];
            for (int j = 0; j < 6; j++) {
                left_angles[j]  = data.joint_pos[j];
                right_angles[j] = data.joint_pos[6 + j];
            }

            auto left_params  = isaaclab::fk::kLeftLegParams();
            auto right_params = isaaclab::fk::kRightLegParams();

            auto left_lm  = isaaclab::fk::legFootLandmarks(left_angles, left_params.data());
            auto right_lm = isaaclab::fk::legFootLandmarks(right_angles, right_params.data());

            data.left_toe_pos_body   = left_lm.toe;
            data.right_toe_pos_body  = right_lm.toe;
            data.left_heel_pos_body  = left_lm.heel;
            data.right_heel_pos_body = right_lm.heel;
        }
    }

    LowStatePtr lowstate;
};

}