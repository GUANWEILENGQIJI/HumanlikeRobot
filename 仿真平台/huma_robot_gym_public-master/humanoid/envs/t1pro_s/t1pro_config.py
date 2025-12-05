# Copyright (c) 2024-2024, fudeRobot All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class T1PROCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        frame_stack = 15
        c_frame_stack = 3
        num_actions = 12
        num_single_obs = 5 + 3 + 3 *  num_actions + 6  
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 5 + 3 + 4 *  num_actions + 20
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_envs = 4096
        episode_length_s = 24

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/T1Bot/urdf/t1pro.urdf'
        name = "t1pro_s"
        foot_name = "ankle_pitch"
        knee_name = "knee"
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        collapse_fixed_joints = True
        test_mode = False
        pos_soft_limit_range_scale = 0.9

        class test_gait:
            signal_type = 1
            offset = 0.2
            pp = 0.15
            cycle_time = 1.5
            joint_ids = [0, 6]

        if test_mode:
            fix_base_link = True
        else:
            fix_base_link = False

        hip_y_friction = 0.08
        hip_r_friction = 0.04
        hip_p_friction = 0.08
        knee_friction = 0.085
        ankle_r_friction = 1.0
        ankle_p_friction = 1.0

        # damping
        hip_y_damping = 2.5
        hip_r_damping = 15.
        hip_p_damping = 17.5
        knee_damping = 6.5
        ankle_r_damping = 3.
        ankle_p_damping = 12.

        # armature
        hip_y_armature = 1.3
        hip_r_armature = 2.3
        hip_p_armature = 2.5
        knee_armature = 1.6
        ankle_r_armature = 0.45
        ankle_p_armature = 0.9
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        height = [0, 0.04]
        horizontal_scale = 0.1

    class noise:
        add_noise = True
        noise_increasing_steps = 5000

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.2
            ang_vel = 0.1
            lin_vel = 0.1
            imu = 0.05

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.91]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'leg_left_hip_yaw_joint': 0.,    # 0
            'leg_left_hip_roll_joint': 0.,    # 1
            'leg_left_hip_pitch_joint': 0.,    # 2
            'leg_left_knee_joint': 0.,    # 3
            'leg_left_ankle_roll_joint': 0.,  # 4
            'leg_left_ankle_pitch_joint': 0.,  # 5
            'leg_right_hip_yaw_joint': 0.,   # 6
            'leg_right_hip_roll_joint': 0.,   # 7
            'leg_right_hip_pitch_joint': 0.,   # 8
            'leg_right_knee_joint': 0.,   # 9
            'leg_right_ankle_roll_joint': 0.,   # 10
            'leg_right_ankle_pitch_joint': 0.,   # 11
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 200.0, 'hip_roll': 325.0, 'hip_pitch': 1450.0,
                     'knee': 550.0, 'ankle_roll': 130, 'ankle_pitch': 310.,
                    }
        damping = {'hip_yaw': 7.5, 'hip_roll': 12., 'hip_pitch': 75.,
                   'knee': 20., 'ankle_roll': 3., 'ankle_pitch': 3.5,
                   }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_phase = False
        
        randomize_friction = True
        friction_range = [0.2, 2.0]
        
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        
        randomize_dof_friction = True
        dof_friction_range = [-0.2, 0.2]
        
        randomize_dof_damping = True
        dof_damping_range = [-0.2, 0.2]
        
        randomize_dof_armature = True
        dof_armature_range = [-0.2, 0.2]
        
        randomize_cycle_time = True
        use_command_cycle_time = False
        cycle_time_range = [0.8, 0.8]
        
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]   
        
        randomize_calculated_torque = True
        torque_multiplier_range = [0.8, 1.2]
        
        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.2]
        
        
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles
        
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 10]
        
        add_obs_latency = True # no latency for obs_action
        randomize_obs_motor_latency = True
        randomize_obs_imu_latency = True
        range_obs_motor_latency = [1, 10]
        range_obs_imu_latency = [1, 10]
        
        
        push_robots = True
        push_interval_s = 12  # 4
        max_push_vel_xy = 0.6  # 0.4 # 2.0
        max_push_ang_vel = 0.3
        push_duration = 1  # 1
        push_curriculum_start_step = 2000 * 60
        push_curriculum_common_step = 10000 * 60
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

        randomize_body_com = True
        added_body_com_range = [-0.06, 0.06]  # 1cm -> 0.02
        added_leg_com_range = [-0.05, 0.05]  # 0.5cm -> 0.005

        randomize_body_inertia = True
        scaled_body_inertia_range = [0.90, 1.1]  # %5 error

        default_joint_angles_range = {  # = target angles [rad] when action = 0.0
            'leg_left_hip_yaw_joint': [-0.1, 0.1],    # 0
            'leg_left_hip_roll_joint':  [-0.1, 0.1],    # 1
            'leg_left_hip_pitch_joint':  [-0.1, 0.1],    # 2
            'leg_left_knee_joint':  [-0.1, 0.1],    # 3
            'leg_left_ankle_roll_joint':  [-0.1, 0.1],  # 4
            'leg_left_ankle_pitch_joint':  [-0.1, 0.1],  # 5

            'leg_right_hip_yaw_joint':  [-0.1, 0.1],   # 6
            'leg_right_hip_roll_joint':  [-0.1, 0.1],   # 7
            'leg_right_hip_pitch_joint':  [-0.1, 0.1],   # 8
            'leg_right_knee_joint':  [-0.1, 0.1],   # 9
            'leg_right_ankle_roll_joint':  [-0.1, 0.1],  # 10
            'leg_right_ankle_pitch_joint':  [-0.1, 0.1],  # 11
        }

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 3
        resampling_time = 8.  # time before command are changed[s]
        max_curriculum = 1.
        curriculum = True
        class ranges:
            lin_vel_x = [-0.5, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]

    class rewards:
        base_height_target = 0.87
        min_dist = 0.16
        max_dist = 1.0
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale_knee = 0.6    # rad 0.25 0.2
        target_joint_pos_scale = 0.35    # rad 0.25 0.2
        target_feet_height = 0.04
        cycle_time = 0.8                # sec [0.32-0.64]
        only_positive_rewards = True
        tracking_sigma = 5
        max_contact_force = 500  # forces above this value are penalized
        double_support_threshold = 0.3
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        class scales:
            # gait
            joint_pos = 1.6
            feet_contact_number = 1.2
            # contact
            feet_contact_forces = -0.002
            # vel tracking
            tracking_lin_vel = 1.8
            tracking_ang_vel = 1.2
            # base pos
            orientation = 1.0
            # energy
            action_smoothness = -0.003
            torques = -1e-7
            dof_vel = -5e-6
            dof_acc = -1e-7
            joint_power = -2e-5
            collision = -1.
            # stand
            zero_stand = 2.0
            # foot
            feet_orientation = 0.5
            foot_slip = -0.01
            feet_distance = 0.2
            feet_clearance = 1.5
            # leg
            default_joint_pos = 1.
            dof_pos_limits = -10.0
            torques_limits = -0.001
            feet_stumble = -1.25
            knee_torques = 1.0
            hipr_torques = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 1.  # cmd_x/y_linear_scale torso_linear_vel_scale
            ang_vel = 1.   # imu_angle_vel_scale torso_ang_vel_scale
            dof_pos = 1.   # joint position scale
            dof_vel = 1.  # joint velocities scale
            imu = 1.0      # imu orientation scale  default 1
        clip_observations = 100.
        clip_actions = 100.

class T1PROCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner
    num_steps_per_env = 60  # per iteration
    save_interval = 100  # check for potential saves every this many iterations
    empirical_normalization = False
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        class_name = 'ActorCritic'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.95
        num_mini_batches = 4
        class_name = 'PPO'

    class runner:
        max_iterations = 20001  # number of policy updates
        # logging
        experiment_name = 't1pro_s'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt