import argparse
import json
import time
from pathlib import Path


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "T1Bot" / "mjcf" / "t1pro2.xml"

PRESETS = {
    "zero": {},
    "stand": {
        "leg_left_hip_pitch_joint": -0.28,
        "leg_right_hip_pitch_joint": -0.28,
        "leg_left_knee_joint": 0.62,
        "leg_right_knee_joint": 0.62,
        "leg_left_ankle_pitch_joint": -0.34,
        "leg_right_ankle_pitch_joint": -0.34,
        "arm_left_shoulder_hunch_joint": 0.10,
        "arm_right_shoulder_hunch_joint": -0.10,
        "arm_left_shoulder_pitch_joint": 0.25,
        "arm_right_shoulder_pitch_joint": 0.25,
        "arm_left_shoulder_roll_joint": 0.18,
        "arm_right_shoulder_roll_joint": 0.18,
        "arm_left_elbow_joint": 0.45,
        "arm_right_elbow_joint": 0.45,
    },
    "squat": {
        "leg_left_hip_pitch_joint": -0.55,
        "leg_right_hip_pitch_joint": -0.55,
        "leg_left_knee_joint": 1.15,
        "leg_right_knee_joint": 1.15,
        "leg_left_ankle_pitch_joint": -0.55,
        "leg_right_ankle_pitch_joint": -0.55,
        "torso_pitch_joint": 0.18,
        "arm_left_shoulder_pitch_joint": 0.35,
        "arm_right_shoulder_pitch_joint": 0.35,
    },
    "arms_forward": {
        "arm_left_shoulder_pitch_joint": 1.0,
        "arm_right_shoulder_pitch_joint": 1.0,
        "arm_left_shoulder_roll_joint": 0.20,
        "arm_right_shoulder_roll_joint": 0.20,
        "arm_left_elbow_joint": 0.50,
        "arm_right_elbow_joint": 0.50,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize T1Pro initial pose in MuJoCo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to T1Pro MJCF model.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="stand", help="Built-in pose preset.")
    parser.add_argument("--pose-file", type=Path, default=None, help="JSON file containing {joint_name: value}.")
    parser.add_argument("--set", dest="sets", action="append", default=[], help="Override with joint=value. Repeatable.")
    parser.add_argument("--base-z", type=float, default=None, help="Override floating base height.")
    parser.add_argument("--watch", action="store_true", help="Watch pose file and hot-reload while viewer is open.")
    parser.add_argument("--print-joints", action="store_true", help="Print joint names and limits, then continue.")
    parser.add_argument("--dump-template", type=Path, default=None, help="Write current preset pose to JSON and exit.")
    return parser.parse_args()


def load_pose_file(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def parse_set_items(items: list[str]) -> dict[str, float]:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value: {item!r}. Expected joint=value.")
        joint, value = item.split("=", 1)
        overrides[joint.strip()] = float(value)
    return overrides


def build_pose(args) -> dict[str, float]:
    pose = dict(PRESETS[args.preset])
    pose.update(load_pose_file(args.pose_file))
    pose.update(parse_set_items(args.sets))
    return pose


def main():
    args = parse_args()

    if args.dump_template is not None:
        pose = build_pose(args)
        args.dump_template.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_template.open("w", encoding="utf-8") as f:
            json.dump(pose, f, indent=2, sort_keys=True)
        print(f"Saved pose template to {args.dump_template}")
        return

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    joint_to_qpos = {}
    joint_limits = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name or model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_to_qpos[name] = int(model.jnt_qposadr[i])
        joint_limits[name] = (float(model.jnt_range[i][0]), float(model.jnt_range[i][1]))

    if args.print_joints:
        print("Joint limits:")
        for name, (lower, upper) in joint_limits.items():
            print(f"  {name}: [{lower:.4f}, {upper:.4f}]")

    pose = build_pose(args)
    last_mtime = None

    def apply_pose(pose_map: dict[str, float]):
        data.qpos[:] = 0.0
        if model.nq >= 7:
            data.qpos[3] = 1.0
        if args.base_z is not None and model.nq >= 3:
            data.qpos[2] = args.base_z
        for joint_name, value in pose_map.items():
            if joint_name not in joint_to_qpos:
                print(f"[warn] Unknown joint: {joint_name}")
                continue
            lower, upper = joint_limits[joint_name]
            clipped = min(max(value, lower), upper)
            if clipped != value:
                print(f"[warn] {joint_name} clipped from {value:.4f} to {clipped:.4f}")
            data.qpos[joint_to_qpos[joint_name]] = clipped
        mujoco.mj_forward(model, data)

    apply_pose(pose)

    print("Model:", args.model)
    print("Preset:", args.preset)
    if args.pose_file is not None:
        print("Pose file:", args.pose_file)
        if args.watch:
            print("Watch mode enabled. Edit the JSON file and save to refresh the pose.")
    print("Use --print-joints to inspect all joint names and limits.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if args.watch and args.pose_file is not None and args.pose_file.exists():
                mtime = args.pose_file.stat().st_mtime
                if last_mtime is None or mtime > last_mtime:
                    last_mtime = mtime
                    try:
                        pose = build_pose(args)
                        apply_pose(pose)
                        print(f"Reloaded pose from {args.pose_file}")
                    except Exception as exc:
                        print(f"[warn] Failed to reload pose: {exc}")
            viewer.sync()
            time.sleep(0.03)


if __name__ == "__main__":
    main()
