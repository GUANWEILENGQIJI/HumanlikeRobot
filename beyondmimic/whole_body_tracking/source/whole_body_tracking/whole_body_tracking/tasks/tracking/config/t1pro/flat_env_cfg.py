from isaaclab.utils import configclass

from whole_body_tracking.robots.t1pro import T1PRO_ACTION_SCALE, T1PRO_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class T1ProFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T1PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1PRO_ACTION_SCALE

        self.commands.motion.anchor_body_name = "torso_pitch_Link"
        self.commands.motion.body_names = [
            "base_link",
            "leg_left_hip_roll_Link",
            "leg_left_knee_Link",
            "leg_left_ankle_pitch_Link",
            "leg_right_hip_roll_Link",
            "leg_right_knee_Link",
            "leg_right_ankle_pitch_Link",
            "torso_pitch_Link",
            "arm_left_shoulder_roll_Link",
            "arm_left_elbow_Link",
            "arm_left_wrist_pitch_Link",
            "arm_right_shoulder_roll_Link",
            "arm_right_elbow_Link",
            "arm_right_wrist_pitch_Link",
        ]

        self.events.base_com.params["asset_cfg"].body_names = "torso_pitch_Link"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!leg_left_ankle_pitch_Link$)(?!leg_right_ankle_pitch_Link$)(?!arm_left_wrist_pitch_Link$)(?!arm_right_wrist_pitch_Link$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "leg_left_ankle_pitch_Link",
            "leg_right_ankle_pitch_Link",
            "arm_left_wrist_pitch_Link",
            "arm_right_wrist_pitch_Link",
        ]

        self.viewer.eye = (2.0, 2.0, 1.8)
        self.viewer.lookat = (0.0, 0.0, 1.0)
