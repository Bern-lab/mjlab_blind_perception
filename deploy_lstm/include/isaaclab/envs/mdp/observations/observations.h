// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    auto cmd = isaaclab::mdp::velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}

// ────────────────────────────────────────────────────────────────────────
//  stair_latent  observation  (91‑dim deployable proprioceptive features)
// ────────────────────────────────────────────────────────────────────────
//
//  Feature order MUST match
//  src/mjlab/tasks/velocity/mdp/observations.py :: stair_latent_obs() :
//
//     1. projected_gravity            (3)
//     2. base_ang_vel                 (3)
//     3. base_ang_vel_delta           (3)
//     4. left_toe_pos_body            (3)
//     5. right_toe_pos_body           (3)
//     6. left_heel_pos_body           (3)
//     7. right_heel_pos_body          (3)
//     8. toe_delta_pos   = L-R       (3)
//     9. heel_delta_pos  = L-R       (3)
//    10. toe_horizontal_distance      (1)
//    11. toe_vertical_distance        (1)
//    12. heel_vertical_distance       (1)
//    13. left_toe_vel_body            (3)
//    14. right_toe_vel_body           (3)
//    15. left_toe_vel_delta           (3)
//    16. right_toe_vel_delta          (3)
//    17. previous_action_leg          (12)
//    18. leg_joint_tracking_error     (12)
//    19. leg_joint_vel                (12)
//    20. leg_joint_vel_delta          (12)
//    21. command_lin_x                (1)
//  ─────────────────────────────────────────
//  Total: 91
// ────────────────────────────────────────────────────────────────────────

REGISTER_OBSERVATION(stair_latent)
{
    auto& asset = env->robot;
    auto& data  = asset->data;
    float dt    = env->step_dt;

    // ── helper: load action scale / offset for the 12 leg joints ──
    auto get_leg_action_scale_offset = [&](std::vector<float>& scale_out,
                                           std::vector<float>& offset_out) {
        scale_out.resize(12, 1.0f);
        offset_out.resize(12, 0.0f);
        try {
            auto actions_node = env->cfg["actions"];
            auto term_it = actions_node.begin();
            if (term_it != actions_node.end()) {
                auto action_cfg = term_it->second;
                if (!action_cfg["scale"].IsNull()) {
                    auto s = action_cfg["scale"].as<std::vector<float>>();
                    for (int i = 0; i < 12 && i < (int)s.size(); ++i)
                        scale_out[i] = s[i];
                }
                if (!action_cfg["offset"].IsNull()) {
                    auto o = action_cfg["offset"].as<std::vector<float>>();
                    for (int i = 0; i < 12 && i < (int)o.size(); ++i)
                        offset_out[i] = o[i];
                }
            }
        } catch (...) {}
    };

    // ── helper: append a 3-vector ──
    auto push3 = [](std::vector<float>& v, const Eigen::Vector3f& e) {
        v.push_back(e.x()); v.push_back(e.y()); v.push_back(e.z());
    };

    // ── 1. projected_gravity (3) ──
    std::vector<float> pg(3);
    pg[0] = data.projected_gravity_b.x();
    pg[1] = data.projected_gravity_b.y();
    pg[2] = data.projected_gravity_b.z();

    // ── 2. base_ang_vel (3) ──
    std::vector<float> bav(3);
    bav[0] = data.root_ang_vel_b.x();
    bav[1] = data.root_ang_vel_b.y();
    bav[2] = data.root_ang_vel_b.z();

    // ── 3. base_ang_vel_delta (3) ──
    Eigen::Vector3f bav_delta = Eigen::Vector3f::Zero();
    if (data.stair_latent_cache_valid) {
        bav_delta = data.root_ang_vel_b - data.prev_base_ang_vel;
    }
    data.prev_base_ang_vel = data.root_ang_vel_b;

    // ── 4-7. Foot FK positions (12 floats) ──
    Eigen::Vector3f l_toe  = data.left_toe_pos_body;
    Eigen::Vector3f r_toe  = data.right_toe_pos_body;
    Eigen::Vector3f l_heel = data.left_heel_pos_body;
    Eigen::Vector3f r_heel = data.right_heel_pos_body;

    // ── 8-12. Foot relative geometry (9 floats) ──
    Eigen::Vector3f toe_delta  = l_toe - r_toe;
    Eigen::Vector3f heel_delta = l_heel - r_heel;
    float toe_h_dist = std::sqrt(toe_delta.x() * toe_delta.x() +
                                 toe_delta.y() * toe_delta.y());
    float toe_v_dist = toe_delta.z();
    float heel_v_dist = heel_delta.z();

    // ── 13-14. Toe velocities (6 floats, finite differences) ──
    Eigen::Vector3f l_toe_vel = Eigen::Vector3f::Zero();
    Eigen::Vector3f r_toe_vel = Eigen::Vector3f::Zero();
    if (data.stair_latent_cache_valid) {
        l_toe_vel = (l_toe - data.prev_left_toe_pos_body) / dt;
        r_toe_vel = (r_toe - data.prev_right_toe_pos_body) / dt;
    }
    data.prev_left_toe_pos_body  = l_toe;
    data.prev_right_toe_pos_body = r_toe;

    // ── 15-16. Toe velocity deltas (6 floats) ──
    Eigen::Vector3f l_toe_vel_delta = Eigen::Vector3f::Zero();
    Eigen::Vector3f r_toe_vel_delta = Eigen::Vector3f::Zero();
    {
        if (data.prev_toe_vel_valid && data.stair_latent_cache_valid) {
            l_toe_vel_delta = l_toe_vel - data.prev_left_toe_vel;
            r_toe_vel_delta = r_toe_vel - data.prev_right_toe_vel;
        }
        data.prev_left_toe_vel  = l_toe_vel;
        data.prev_right_toe_vel = r_toe_vel;
        data.prev_toe_vel_valid = true;
    }

    // ── 17. previous_action_leg (12) ──
    // IMPORTANT: At compute() time (before process_action() is called),
    // action_manager->action() returns the PREVIOUS frame's raw ONNX output.
    // This matches Python's env.extras["prev_action"].
    std::vector<float> prev_action_leg(12, 0.0f);
    {
        auto raw_action = env->action_manager->action();
        for (int i = 0; i < 12; ++i)
            prev_action_leg[i] = raw_action[i];
    }

    // ── 18. leg_joint_tracking_error (12) ──
    // Python definition:
    //   target_joint_pos = default_joint_pos + action_scale * previous_action_leg
    //   tracking_error   = target_joint_pos - current_joint_pos
    //                    = (q_def + scale * raw_action) - measured_q
    // NOTE: action_offset in deploy.yaml equals default_joint_pos,
    // but we use the explicit formula above to match training exactly.
    std::vector<float> leg_tracking_error(12, 0.0f);
    {
        std::vector<float> action_scale(12, 1.0f);
        std::vector<float> action_offset(12, 0.0f);
        get_leg_action_scale_offset(action_scale, action_offset);

        for (int i = 0; i < 6; ++i) {
            // left leg: tracking_error = (default + scale * raw_action) - measured
            float q_measured = data.joint_pos[i];
            float q_default  = data.default_joint_pos[i];
            float target     = q_default + prev_action_leg[i] * action_scale[i];
            leg_tracking_error[i] = target - q_measured;
        }
        for (int i = 0; i < 6; ++i) {
            // right leg (joint indices 6..11)
            float q_measured = data.joint_pos[6 + i];
            float q_default  = data.default_joint_pos[6 + i];
            float target     = q_default + prev_action_leg[6 + i] * action_scale[6 + i];
            leg_tracking_error[6 + i] = target - q_measured;
        }
    }

    // ── 19. leg_joint_vel (12) ──
    std::vector<float> leg_joint_vel(12, 0.0f);
    for (int i = 0; i < 6; ++i) {
        leg_joint_vel[i]     = data.joint_vel[i];
        leg_joint_vel[6 + i] = data.joint_vel[6 + i];
    }

    // ── 20. leg_joint_vel_delta (12) ──
    std::vector<float> leg_joint_vel_delta(12, 0.0f);
    {
        // Use prev_leg_joint_vel from ArticulationData.
        // First call: prev_leg_joint_vel is empty → delta = 0.
        if (data.prev_leg_joint_vel.size() == 12 && data.stair_latent_cache_valid) {
            for (int i = 0; i < 12; ++i)
                leg_joint_vel_delta[i] = leg_joint_vel[i] - data.prev_leg_joint_vel[i];
        }
        data.prev_leg_joint_vel.resize(12);
        for (int i = 0; i < 12; ++i)
            data.prev_leg_joint_vel[i] = leg_joint_vel[i];
    }

    // ── 21. command_lin_x (1) ──
    auto& joystick = data.joystick;
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    float cmd_lin_x = std::clamp(joystick->ly(),
                                 cfg["lin_vel_x"][0].as<float>(),
                                 cfg["lin_vel_x"][1].as<float>());

    // ── Mark cache as valid for next call ──
    data.stair_latent_cache_valid = true;

    // ── Assemble final 91‑dim vector ──
    std::vector<float> out;
    out.reserve(91);

    //  1   projected_gravity (3)
    out.insert(out.end(), pg.begin(), pg.end());
    //  2   base_ang_vel (3)
    out.insert(out.end(), bav.begin(), bav.end());
    //  3   base_ang_vel_delta (3)
    push3(out, bav_delta);
    //  4   left_toe_pos_body (3)
    push3(out, l_toe);
    //  5   right_toe_pos_body (3)
    push3(out, r_toe);
    //  6   left_heel_pos_body (3)
    push3(out, l_heel);
    //  7   right_heel_pos_body (3)
    push3(out, r_heel);
    //  8   toe_delta_pos (3)
    push3(out, toe_delta);
    //  9   heel_delta_pos (3)
    push3(out, heel_delta);
    // 10   toe_horizontal_distance (1)
    out.push_back(toe_h_dist);
    // 11   toe_vertical_distance (1)
    out.push_back(toe_v_dist);
    // 12   heel_vertical_distance (1)
    out.push_back(heel_v_dist);
    // 13   left_toe_vel_body (3)
    push3(out, l_toe_vel);
    // 14   right_toe_vel_body (3)
    push3(out, r_toe_vel);
    // 15   left_toe_vel_delta (3)
    push3(out, l_toe_vel_delta);
    // 16   right_toe_vel_delta (3)
    push3(out, r_toe_vel_delta);
    // 17   previous_action_leg (12)
    out.insert(out.end(), prev_action_leg.begin(), prev_action_leg.end());
    // 18   leg_joint_tracking_error (12)
    out.insert(out.end(), leg_tracking_error.begin(), leg_tracking_error.end());
    // 19   leg_joint_vel (12)
    out.insert(out.end(), leg_joint_vel.begin(), leg_joint_vel.end());
    // 20   leg_joint_vel_delta (12)
    out.insert(out.end(), leg_joint_vel_delta.begin(), leg_joint_vel_delta.end());
    // 21   command_lin_x (1)
    out.push_back(cmd_lin_x);

    return out;
}

// ────────────────────────────────────────────────────────────────────────
//  target_heading_commands  observation  (3‑dim, used as
//  "velocity_commands" when the task is target-navigation)
// ────────────────────────────────────────────────────────────────────────
//
//  Falls back to joystick velocity commands (same as velocity_commands)
//  since the real-robot deployment uses joystick control.
//
REGISTER_OBSERVATION(target_heading_commands)
{
    // On real hardware we always use joystick velocity commands;
    // target-heading mode only exists in simulation.
    return isaaclab::mdp::velocity_commands(env, params);
}

}
}
