from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

try:
    from whole_body_tracking.assets import ASSET_DIR as _ASSET_DIR
except ImportError:
    _ASSET_DIR = None


def _resolve_t1pro_urdf() -> str:
    if _ASSET_DIR is not None:
        candidate = Path(_ASSET_DIR) / "T1Bot" / "urdf" / "t1pro.urdf"
        if candidate.exists():
            return str(candidate)
    return str(Path(__file__).resolve().parent / "T1Bot" / "urdf" / "t1pro.urdf")


T1PRO_URDF_PATH = _resolve_t1pro_urdf()

NATURAL_FREQ = 8.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 1.6


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


ARMATURES = {
    "hip_yaw": 1.3,
    "hip_roll": 2.0,
    "hip_pitch": 2.5,
    "knee": 1.4,
    "ankle_yaw": 0.3,
    "ankle_roll": 0.45,
    "ankle_pitch": 0.9,
    "torso_yaw": 0.3,
    "torso_roll": 2.0,
    "torso_pitch": 2.5,
    "arm_hunch": 0.12,
    "arm_shoulder": 0.2,
    "elbow": 0.18,
    "wrist": 0.08,
    "neck": 0.05,
    "head": 0.05,
}


T1PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=T1PRO_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            "leg_.*_hip_pitch_joint": 0.2,
            "leg_.*_knee_joint": 0.62,
            "leg_.*_ankle_pitch_joint": 0.42,
            "arm_left_shoulder_hunch_joint": 0.10,
            "arm_right_shoulder_hunch_joint": -0.10,
            "arm_left_shoulder_pitch_joint": 0.25,
            "arm_right_shoulder_pitch_joint": 0.25,
            "arm_left_shoulder_roll_joint": 0.18,
            "arm_right_shoulder_roll_joint": 0.18,
            "arm_left_elbow_joint": 0.45,
            "arm_right_elbow_joint": 0.45,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*_hip_yaw_joint",
                "leg_.*_hip_roll_joint",
                "leg_.*_hip_pitch_joint",
                "leg_.*_knee_joint",
               
                "leg_.*_ankle_roll_joint",
                "leg_.*_ankle_pitch_joint",
            ],
            effort_limit_sim={
                "leg_.*_hip_yaw_joint": 80.0,
                "leg_.*_hip_roll_joint": 100.0,
                "leg_.*_hip_pitch_joint": 200.0,
                "leg_.*_knee_joint": 150.0,
               
                "leg_.*_ankle_roll_joint": 40.0,
                "leg_.*_ankle_pitch_joint": 80.0,
            },
            velocity_limit_sim={
                "leg_.*_hip_yaw_joint": 12.5664,
                "leg_.*_hip_roll_joint": 12.5664,
                "leg_.*_hip_pitch_joint": 12.5664,
                "leg_.*_knee_joint": 12.5664,
                
                "leg_.*_ankle_roll_joint": 12.5664,
                "leg_.*_ankle_pitch_joint": 12.5664,
            },
            stiffness={
                "leg_.*_hip_yaw_joint": _stiffness(ARMATURES["hip_yaw"]),
                "leg_.*_hip_roll_joint": _stiffness(ARMATURES["hip_roll"]),
                "leg_.*_hip_pitch_joint": _stiffness(ARMATURES["hip_pitch"]),
                "leg_.*_knee_joint": _stiffness(ARMATURES["knee"]),
               
                "leg_.*_ankle_roll_joint": _stiffness(ARMATURES["ankle_roll"]),
                "leg_.*_ankle_pitch_joint": _stiffness(ARMATURES["ankle_pitch"]),
            },
            damping={
                "leg_.*_hip_yaw_joint": _damping(ARMATURES["hip_yaw"]),
                "leg_.*_hip_roll_joint": _damping(ARMATURES["hip_roll"]),
                "leg_.*_hip_pitch_joint": _damping(ARMATURES["hip_pitch"]),
                "leg_.*_knee_joint": _damping(ARMATURES["knee"]),
               
                "leg_.*_ankle_roll_joint": _damping(ARMATURES["ankle_roll"]),
                "leg_.*_ankle_pitch_joint": _damping(ARMATURES["ankle_pitch"]),
            },
            armature={
                "leg_.*_hip_yaw_joint": ARMATURES["hip_yaw"],
                "leg_.*_hip_roll_joint": ARMATURES["hip_roll"],
                "leg_.*_hip_pitch_joint": ARMATURES["hip_pitch"],
                "leg_.*_knee_joint": ARMATURES["knee"],
             
                "leg_.*_ankle_roll_joint": ARMATURES["ankle_roll"],
                "leg_.*_ankle_pitch_joint": ARMATURES["ankle_pitch"],
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "torso_yaw_joint",
                "torso_roll_joint",
                "torso_pitch_joint",
            ],
            effort_limit_sim={
                "torso_yaw_joint": 73.0,
                "torso_roll_joint": 450.0,
                "torso_pitch_joint": 450.0,
            },
            velocity_limit_sim={
                "torso_yaw_joint": 12.5664,
                "torso_roll_joint": 12.5664,
                "torso_pitch_joint": 12.5664,
            },
            stiffness={
                "torso_yaw_joint": _stiffness(ARMATURES["torso_yaw"]),
                "torso_roll_joint": _stiffness(ARMATURES["torso_roll"]),
                "torso_pitch_joint": _stiffness(ARMATURES["torso_pitch"]),
            },
            damping={
                "torso_yaw_joint": _damping(ARMATURES["torso_yaw"]),
                "torso_roll_joint": _damping(ARMATURES["torso_roll"]),
                "torso_pitch_joint": _damping(ARMATURES["torso_pitch"]),
            },
            armature={
                "torso_yaw_joint": ARMATURES["torso_yaw"],
                "torso_roll_joint": ARMATURES["torso_roll"],
                "torso_pitch_joint": ARMATURES["torso_pitch"],
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_.*_shoulder_hunch_joint",
                "arm_.*_shoulder_pitch_joint",
                "arm_.*_shoulder_roll_joint",
                "arm_.*_shoulder_yaw_joint",
                "arm_.*_elbow_joint",
                "arm_.*_wrist_roll_joint",
                "arm_.*_wrist_yaw_joint",
                "arm_.*_wrist_pitch_joint",
            ],
            effort_limit_sim={
                "arm_.*_shoulder_hunch_joint": 30.0,
                "arm_.*_shoulder_pitch_joint": 96.0,
                "arm_.*_shoulder_roll_joint": 96.0,
                "arm_.*_shoulder_yaw_joint": 56.0,
                "arm_.*_elbow_joint": 56.0,
                "arm_.*_wrist_roll_joint": 30.0,
                "arm_.*_wrist_yaw_joint": 30.0,
                "arm_.*_wrist_pitch_joint": 30.0,
            },
            velocity_limit_sim={
                "arm_.*_shoulder_hunch_joint": 7.854,
                "arm_.*_shoulder_pitch_joint": 7.854,
                "arm_.*_shoulder_roll_joint": 7.854,
                "arm_.*_shoulder_yaw_joint": 7.854,
                "arm_.*_elbow_joint": 7.854,
                "arm_.*_wrist_roll_joint": 7.854,
                "arm_.*_wrist_yaw_joint": 7.854,
                "arm_.*_wrist_pitch_joint": 7.854,
            },
            stiffness={
                "arm_.*_shoulder_hunch_joint": _stiffness(ARMATURES["arm_hunch"]),
                "arm_.*_shoulder_pitch_joint": _stiffness(ARMATURES["arm_shoulder"]),
                "arm_.*_shoulder_roll_joint": _stiffness(ARMATURES["arm_shoulder"]),
                "arm_.*_shoulder_yaw_joint": _stiffness(ARMATURES["arm_shoulder"]),
                "arm_.*_elbow_joint": _stiffness(ARMATURES["elbow"]),
                "arm_.*_wrist_roll_joint": _stiffness(ARMATURES["wrist"]),
                "arm_.*_wrist_yaw_joint": _stiffness(ARMATURES["wrist"]),
                "arm_.*_wrist_pitch_joint": _stiffness(ARMATURES["wrist"]),
            },
            damping={
                "arm_.*_shoulder_hunch_joint": _damping(ARMATURES["arm_hunch"]),
                "arm_.*_shoulder_pitch_joint": _damping(ARMATURES["arm_shoulder"]),
                "arm_.*_shoulder_roll_joint": _damping(ARMATURES["arm_shoulder"]),
                "arm_.*_shoulder_yaw_joint": _damping(ARMATURES["arm_shoulder"]),
                "arm_.*_elbow_joint": _damping(ARMATURES["elbow"]),
                "arm_.*_wrist_roll_joint": _damping(ARMATURES["wrist"]),
                "arm_.*_wrist_yaw_joint": _damping(ARMATURES["wrist"]),
                "arm_.*_wrist_pitch_joint": _damping(ARMATURES["wrist"]),
            },
            armature={
                "arm_.*_shoulder_hunch_joint": ARMATURES["arm_hunch"],
                "arm_.*_shoulder_pitch_joint": ARMATURES["arm_shoulder"],
                "arm_.*_shoulder_roll_joint": ARMATURES["arm_shoulder"],
                "arm_.*_shoulder_yaw_joint": ARMATURES["arm_shoulder"],
                "arm_.*_elbow_joint": ARMATURES["elbow"],
                "arm_.*_wrist_roll_joint": ARMATURES["wrist"],
                "arm_.*_wrist_yaw_joint": ARMATURES["wrist"],
                "arm_.*_wrist_pitch_joint": ARMATURES["wrist"],
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "neck_roll_joint",
                "neck_pitch_joint",
                "head_roll_joint",
                "head_pitch_joint",
                "head_yaw_joint",
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=7.854,
            stiffness={
                "neck_roll_joint": _stiffness(ARMATURES["neck"]),
                "neck_pitch_joint": _stiffness(ARMATURES["neck"]),
                "head_roll_joint": _stiffness(ARMATURES["head"]),
                "head_pitch_joint": _stiffness(ARMATURES["head"]),
                "head_yaw_joint": _stiffness(ARMATURES["head"]),
            },
            damping={
                "neck_roll_joint": _damping(ARMATURES["neck"]),
                "neck_pitch_joint": _damping(ARMATURES["neck"]),
                "head_roll_joint": _damping(ARMATURES["head"]),
                "head_pitch_joint": _damping(ARMATURES["head"]),
                "head_yaw_joint": _damping(ARMATURES["head"]),
            },
            armature={
                "neck_roll_joint": ARMATURES["neck"],
                "neck_pitch_joint": ARMATURES["neck"],
                "head_roll_joint": ARMATURES["head"],
                "head_pitch_joint": ARMATURES["head"],
                "head_yaw_joint": ARMATURES["head"],
            },
        ),
    },
)


T1PRO_ACTION_SCALE = {}
for actuator_cfg in T1PRO_CFG.actuators.values():
    effort = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    names = actuator_cfg.joint_names_expr
    if not isinstance(effort, dict):
        effort = {name: effort for name in names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in names}
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            T1PRO_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
