# Copyright (c) 2024-2024, fudeRobot All rights reserved.
from time import time
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import random
import torch
import numpy as np
from humanoid.envs import LeggedRobot
from humanoid.utils.terrain import  Terrain



def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class T1PROEnv(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        push_curriculum_scale = np.clip((self.common_step_counter - self.cfg.domain_rand.push_curriculum_start_step)
                                        / self.cfg.domain_rand.push_curriculum_common_step, 0, 1)
        if push_curriculum_scale > 0.:
            print("------push robot at step:", self.common_step_counter)
        max_vel = self.cfg.domain_rand.max_push_vel_xy * push_curriculum_scale
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel * push_curriculum_scale

        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)
        self.root_states[:, 10:13] = self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _get_phase(self):
        if self.cfg.domain_rand.randomize_cycle_time is True:
            cycle_time = torch.squeeze_copy(self.rand_cycle_time)
        else:
            cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        if self.cfg.domain_rand.randomize_phase is True:
            phase = phase + self.delta_phase
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < self.cfg.rewards.double_support_threshold] = 1

        if self.cfg.rewards.scales.zero_stand != 0:
            if torch.numel(self.stand_idx) != 0:
                stance_mask[self.stand_idx] = 1
        return stance_mask
    
    def compute_ref_state(self):
        phase = self._get_phase()
        _sin_pos_l = torch.sin(2 * torch.pi * phase)
        _sin_pos_r = torch.sin(2 * torch.pi * phase + torch.pi)
        sin_pos_l = _sin_pos_l.clone()
        sin_pos_r = _sin_pos_r.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        ratio_l = torch.clamp(torch.abs(sin_pos_l) - self.cfg.rewards.double_support_threshold, min=0, max=1) / (1 - self.cfg.rewards.double_support_threshold) * torch.sign(sin_pos_l)
        self.ref_dof_pos[:, 2] = -ratio_l * scale_1
        self.ref_dof_pos[:, 3] = -ratio_l * scale_2
        self.ref_dof_pos[:, 5] = -ratio_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r > 0] = 0
        ratio_r = torch.clamp(torch.abs(sin_pos_r) - self.cfg.rewards.double_support_threshold, min=0, max=1) / (1 - self.cfg.rewards.double_support_threshold) * torch.sign(sin_pos_r)
        self.ref_dof_pos[:, 8] = -ratio_r * scale_1
        self.ref_dof_pos[:, 9] = -ratio_r * scale_2
        self.ref_dof_pos[:, 11] = -ratio_r * scale_1
        # Double support phase
        indices = (torch.abs(sin_pos_l) < self.cfg.rewards.double_support_threshold) & (torch.abs(sin_pos_r) < self.cfg.rewards.double_support_threshold)
        self.ref_dof_pos[indices] = 0
        # self.ref_dof_pos += self.default_dof_pos_all
        if self.cfg.rewards.scales.zero_stand != 0:
            if torch.numel(self.stand_idx) != 0:
                self.ref_dof_pos[self.stand_idx] = 0
        self.ref_action = 2 * self.ref_dof_pos
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 8] = 0.  # commands
        noise_vec[8: 8 + self.num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[8 + self.num_actions: 8 + 2 * self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[8 + 2 * self.num_actions: 8 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[8 + 3 * self.num_actions: 8 + 3 * self.num_actions + 3] = \
            noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[8 + 3 * self.num_actions + 3: 8 + 3 * self.num_actions + 6] = noise_scales.imu * self.obs_scales.imu
        return noise_vec

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self._post_physics_step_callback()

        self.stand_idx = torch.where((torch.abs(self.commands[:, 0])  < 0.1)
                                     & (torch.abs(self.commands[:, 1]) < 0.1)
                                     & (torch.abs(self.commands[:, 2]) < 0.1)
                                     & (self.stand_flag[:, 0] == 1.))[0]

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit

        # friction randomization
        if self.cfg.domain_rand.randomize_dof_friction and (self.cfg.asset.test_mode is False):
            rg = self.cfg.domain_rand.dof_friction_range
            hip_y_friction = self.cfg.asset.hip_y_friction * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            hip_r_friction = self.cfg.asset.hip_r_friction * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            hip_p_friction = self.cfg.asset.hip_p_friction * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            knee_friction = self.cfg.asset.knee_friction * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            ankle_r_friction = self.cfg.asset.ankle_r_friction * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            ankle_p_friction = self.cfg.asset.ankle_p_friction * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
        else:
            hip_y_friction = self.cfg.asset.hip_y_friction
            hip_r_friction = self.cfg.asset.hip_r_friction
            hip_p_friction = self.cfg.asset.hip_p_friction
            knee_friction = self.cfg.asset.knee_friction
            ankle_r_friction = self.cfg.asset.ankle_r_friction
            ankle_p_friction = self.cfg.asset.ankle_p_friction
            
        # damping randomization
        if self.cfg.domain_rand.randomize_dof_damping and (self.cfg.asset.test_mode is False):
            rg = self.cfg.domain_rand.dof_damping_range
            hip_y_damping = self.cfg.asset.hip_y_damping * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            hip_r_damping = self.cfg.asset.hip_r_damping * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            hip_p_damping = self.cfg.asset.hip_p_damping * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            knee_damping = self.cfg.asset.knee_damping * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            ankle_r_damping = self.cfg.asset.ankle_r_damping * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            ankle_p_damping = self.cfg.asset.ankle_p_damping * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
        else:
            hip_y_damping = self.cfg.asset.hip_y_damping
            hip_r_damping = self.cfg.asset.hip_r_damping
            hip_p_damping = self.cfg.asset.hip_p_damping
            knee_damping = self.cfg.asset.knee_damping
            ankle_r_damping = self.cfg.asset.ankle_r_damping
            ankle_p_damping = self.cfg.asset.ankle_p_damping

        # armature randomization
        if self.cfg.domain_rand.randomize_dof_armature and (self.cfg.asset.test_mode is False):
            rg = self.cfg.domain_rand.dof_armature_range
            hip_y_armature = self.cfg.asset.hip_y_armature * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            hip_r_armature = self.cfg.asset.hip_r_armature * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            hip_p_armature = self.cfg.asset.hip_p_armature * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            knee_armature = self.cfg.asset.knee_armature * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            ankle_r_armature = self.cfg.asset.ankle_r_armature * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
            ankle_p_armature = self.cfg.asset.ankle_p_armature * (1 + torch_rand_float(rg[0], rg[1], (1, 1), device='cpu'))
        else:
            hip_y_armature = self.cfg.asset.hip_y_armature
            hip_r_armature = self.cfg.asset.hip_r_armature
            hip_p_armature = self.cfg.asset.hip_p_armature
            knee_armature = self.cfg.asset.knee_armature
            ankle_r_armature = self.cfg.asset.ankle_r_armature
            ankle_p_armature = self.cfg.asset.ankle_p_armature

        # hip_y
        props["friction"][0] = hip_y_friction
        props["friction"][6] = hip_y_friction
        # hip_r
        props["friction"][1] = hip_r_friction
        props["friction"][7] = hip_r_friction
        # hip_p
        props["friction"][2] = hip_p_friction
        props["friction"][8] = hip_p_friction
        # knee
        props["friction"][3] = knee_friction
        props["friction"][9] = knee_friction
        # ankle_r
        props["friction"][4] = ankle_r_friction
        props["friction"][10] = ankle_r_friction
        # ankle_p
        props["friction"][5] = ankle_p_friction
        props["friction"][11] = ankle_p_friction

        # hip_y
        props["damping"][0] = hip_y_damping
        props["damping"][6] = hip_y_damping
        # hip_r
        props["damping"][1] = hip_r_damping
        props["damping"][7] = hip_r_damping
        # hip_p
        props["damping"][2] = hip_p_damping
        props["damping"][8] = hip_p_damping
        # knee
        props["damping"][3] = knee_damping
        props["damping"][9] = knee_damping
        # ankle_r
        props["damping"][4] = ankle_r_damping
        props["damping"][10] = ankle_r_damping
        # ankle_p
        props["damping"][5] = ankle_p_damping
        props["damping"][11] = ankle_p_damping

        # hip_y
        props["armature"][0] = hip_y_armature
        props["armature"][6] = hip_y_armature
        # hip_r
        props["armature"][1] = hip_r_armature
        props["armature"][7] = hip_r_armature
        # hip_p
        props["armature"][2] = hip_p_armature
        props["armature"][8] = hip_p_armature
        # knee
        props["armature"][3] = knee_armature
        props["armature"][9] = knee_armature
        # ankle_r
        props["armature"][4] = ankle_r_armature
        props["armature"][10] = ankle_r_armature
        # ankle_p
        props["armature"][5] = ankle_p_armature
        props["armature"][11] = ankle_p_armature

        return props

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos
        
        add_command = torch.zeros(self.num_envs, 3, device=self.device)
        add_command[:, 0] = self.stand_flag[:, 0].clone()
        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            add_command,
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.imu,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)

        if self.cfg.domain_rand.randomize_obs_motor_latency:
            self.obs_motor = self.obs_motor_latency_buffer[torch.arange(self.num_envs), :, self.obs_motor_latency_simstep.long()]
        else:
            self.obs_motor = torch.cat((q, dq), 1)

        if self.cfg.domain_rand.randomize_obs_imu_latency:
            self.obs_imu = self.obs_imu_latency_buffer[torch.arange(self.num_envs), :, self.obs_imu_latency_simstep.long()]
        else:              
            self.obs_imu = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.base_euler_xyz * self.obs_scales.imu), 1)

        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            add_command,
            self.obs_motor,
            self.actions,   # 12D
            self.obs_imu,
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * min(self.common_step_counter / (self.cfg.noise.noise_increasing_steps * 60),  1.)
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_ankle_torques(self):
        """
            踝关节能量越小越好
        """
        return (torch.exp(-torch.norm(self.torques[:, 4:6], dim=1) * 0.05) + torch.exp(-torch.norm(self.torques[:, 10:12], dim=1) * 0.05)) / 2.0

    def _reward_hipyr_torques(self):
        """
            髋关节rp能量越小越好
        """
        return (torch.exp(-torch.norm(self.torques[:, 0:2], dim=1) * 0.05) + torch.exp(-torch.norm(self.torques[:, 6:8], dim=1) * 0.05)) / 2.0

    def _reward_feet_orientation(self):
        """
            踝关节与地面平行
        """
        feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        footl_euler_xyz = get_euler_xyz_tensor(feet_quat[:, 0])
        footr_euler_xyz = get_euler_xyz_tensor(feet_quat[:, 1])
        footl_rew = torch.exp(-torch.sum(torch.abs(footl_euler_xyz[:, :2]), dim=1) * 10)
        footr_rew = torch.exp(-torch.sum(torch.abs(footr_euler_xyz[:, :2]), dim=1) * 10)
        return (footl_rew + footr_rew) / 2.0

    def _reward_feet_clearance(self):
        """
            抬腿高度
        """
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        stance_mask = self._get_gait_phase()
        phase = self._get_phase()
        swing_mask = 1 - stance_mask

        phase %= 0.5

        body_tilt_ang = (torch.norm(self.base_euler_xyz[:, :2], dim=1).clamp(0.03, 0.3) - 0.03) / (0.3 - 0.03)  # -> [0, 1]
        tilt_ang_height = 0.2 * body_tilt_ang  # max height 0.2
        tilt_feet_height = torch.where(body_tilt_ang > 0.0001, tilt_ang_height + self.cfg.rewards.target_feet_height,
                                       self.cfg.rewards.target_feet_height)
        #feet_height = torch.max(torch.tensor(self.cfg.rewards.target_feet_height), torch.norm(self.commands[:, :2], dim=1) / 10.0)
        feet_height = torch.max(tilt_feet_height, torch.norm(self.commands[:, :2], dim=1) / 10.0)
        feet_height = torch.max(feet_height, torch.norm(self.base_lin_vel[:, :2], dim=1) / 10.0)
        target_feet_height = torch.zeros_like(feet_z, device=self.device)
        target_feet_height[:, 0] = swing_mask[:, 0] * (1 - torch.square(phase - 0.25) / 0.0625) * feet_height
        target_feet_height[:, 1] = swing_mask[:, 1] * (1 - torch.square(phase - 0.25) / 0.0625) * feet_height

        rew_pos = torch.abs(feet_z - target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)

        return rew_pos

    def _reward_default_joint_pos(self):
        """
            髋关节越小越好
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

        rw = torch.exp(-yaw_roll * 100)

        body_tilt_ang = (torch.norm(self.base_euler_xyz[:, :2], dim=1).clamp(0.03, 0.3) - 0.03) / (0.3 - 0.03)  # -> [0, 1]
        rw = torch.where(body_tilt_ang < 0.1, rw, 1)
        return rw

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        pos_limits = self.dof_pos_limits.clone()
        pos_limits[3, 0] = 0.
        pos_limits[9, 0] = 0.
        out_of_limits = -(self.dof_pos - pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_actions_limits(self):
        # Penalize action too close to the limit
        target_action = self.actions * self.cfg.control.action_scale
        action_limit = self.dof_pos_limits.clone()
        action_limit[3, 0] = 0.
        action_limit[9, 0] = 0.
        out_of_limits = -(target_action - action_limit[:, 0]*1.1).clip(max=0.)  # lower limit
        out_of_limits += (target_action - action_limit[:, 1]*1.1).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_feet_stumble(self):
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()
    
    def _reward_feet_contact_forces(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < self.cfg.rewards.max_contact_force] = 0
        rew[rew > self.cfg.rewards.max_contact_force] -= self.cfg.rewards.max_contact_force
        rew[~self.get_walking_cmd_mask()] = 0
        return rew
    
    def get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > 0.1
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > 0.1
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > 0.1
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask
    
    def _reward_knee_torques(self):
        """
            踝关节能量越小越好
        """
        return (torch.exp(-torch.abs(self.torques[:, 3]) * 0.05) + torch.exp(-torch.abs(self.torques[:, 9]) * 0.05)) / 2.0
    
    def _reward_hipr_torques(self):
        """
            踝关节能量越小越好
        """
        return (torch.exp(-torch.abs(self.torques[:, 1]) * 0.05) + torch.exp(-torch.abs(self.torques[:, 7]) * 0.05)) / 2.0
    
    def _reward_torques_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.raw_torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    
    def _reward_zero_stand(self):
        """
        速度为0时, 关节角度值小于0.01奖励, 大于0.01施加惩罚
        """
        reward = torch.zeros_like(self.dof_pos[:, 0])
        if torch.numel(self.stand_idx) != 0:
            dof_sum = torch.sum(torch.square(self.dof_pos[self.stand_idx]), dim=1)
            reward[self.stand_idx] = torch.exp(-dof_sum * 50)
        return reward
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)