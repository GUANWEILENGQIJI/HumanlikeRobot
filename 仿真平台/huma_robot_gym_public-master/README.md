## Installation

1. Generate a new Python virtual environment with Python 3.8 using `conda create -n myenv python=3.8`.
2. For the best performance, we recommend using NVIDIA driver version 525 `sudo apt install nvidia-driver-525`. The minimal driver version supported is 515. If you're unable to install version 525, ensure that your system has at least version 515 to maintain basic functionality.
3. Install PyTorch 1.13 with Cuda-11.7:
   - `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
4. Install numpy-1.23 with `conda install numpy=1.23`.
5. Install Isaac Gym:
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym.
   - `cd isaacgym/python && pip install -e .`
   - Run an example with `cd examples && python 1080_balls_of_solitude.py`.
   - Consult `isaacgym/docs/index.html` for troubleshooting.
6. Install huma_robot_gym:
   - Clone this repository.
   - `cd huma_robot_gym && pip install -e .`


## Usage Guide

#### Examples

```bash
# Under the directory huma_robot_gym/humanoid
python scripts/train.py --task=t1pro_s_ppo--headless --num_envs 4096

# Evaluating the Trained PPO Policy 
python scripts/play.py  --task=t1pro_s_ppo --num_envs 1 

# Implementing Simulation-to-Simulation Model Transformation
python scripts/fdsim2ros.py 
```

#### 1. Default Tasks

- **t1pro_s_ppo**
   - Purpose: Baseline, PPO policy, Multi-frame low-level control

#### 2. PPO Policy
- **Training Command**: For training the PPO policy, execute:
  ```
  python humanoid/scripts/train.py --task=t1pro_s_ppo --headless --num_envs 4096
  ```
- **Running a Trained Policy**: To deploy a trained PPO policy, use:
  ```
  python humanoid/scripts/play.py --task=t1pro_s_ppo 
  ```
- By default, the latest model of the last run from the experiment folder is loaded. However, other run iterations/models can be selected by adjusting `load_run` and `checkpoint` in the training config.

#### 3. Sim-to-sim
- **Please note: Before initiating the sim-to-sim process, ensure that you run `play.py` to export a JIT policy.**
- **Mujoco-based Sim2Sim Deployment**: Utilize Mujoco for executing simulation-to-simulation (sim2sim) deployments with the command below:
  ```
  python scripts/fdsim2ros.py
  ```


#### 4. Parameters
- **CPU and GPU Usage**: To run simulations on the CPU, set both `--sim_device=cpu` and `--rl_device=cpu`. For GPU operations, specify `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}` accordingly. Please note that `CUDA_VISIBLE_DEVICES` is not applicable, and it's essential to match the `--sim_device` and `--rl_device` settings.
- **Headless Operation**: Include `--headless` for operations without rendering.
- **Rendering Control**: Press 'v' to toggle rendering during training.
- **Policy Location**: Trained policies are saved in `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.

## Code Structure

1. Every environment hinges on an `env` file (`legged_robot.py`) and a `configuration` file (`legged_robot_config.py`). The latter houses two classes: `LeggedRobotCfg` (encompassing all environmental parameters) and `LeggedRobotCfgPPO` (denoting all training parameters).
2. Both `env` and `config` classes use inheritance.
3. Non-zero reward scales specified in `cfg` contribute a function of the corresponding name to the sum-total reward.
4. Tasks must be registered with `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. Registration may occur within `envs/__init__.py`, or outside of this repository.


## Add a new environment 

The base environment `legged_robot` constructs a rough terrain locomotion task. The corresponding configuration does not specify a robot asset (URDF/ MJCF) and no reward scales.

1. If you need to add a new environment, create a new folder in the `envs/` directory with a configuration file named `<your_env>_config.py`. The new configuration should inherit from existing environment configurations.
2. If proposing a new robot:
    - Insert the corresponding assets in the `resources/` folder.
    - In the `cfg` file, set the path to the asset, define body names, default_joint_positions, and PD gains. Specify the desired `train_cfg` and the environment's name (python class).
    - In the `train_cfg`, set the `experiment_name` and `run_name`.
3. If needed, create your environment in `<your_env>.py`. Inherit from existing environments, override desired functions and/or add your reward functions.
4. Register your environment in `humanoid/envs/__init__.py`.
5. Modify or tune other parameters in your `cfg` or `cfg_train` as per requirements. To remove the reward, set its scale to zero. Avoid modifying the parameters of other environments!
6. If you want a new robot/environment to perform sim2sim, you may need to modify `humanoid/scripts/fdsim2ros.py`: 
    - Check the joint mapping of the robot between MJCF and URDF.
    - Change the initial joint position of the robot according to your trained policy.

## Troubleshooting

Observe the following cases:

```bash
# error
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

# solution
# set the correct path
export LD_LIBRARY_PATH="~/miniconda3/envs/your_env/lib:$LD_LIBRARY_PATH" 

# OR
sudo apt install libpython3.8

# error
AttributeError: module 'distutils' has no attribute 'version'

# solution
# install pytorch 1.12.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# error, results from libstdc++ version distributed with conda differing from the one used on your system to build Isaac Gym
ImportError: /home/roboterax/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20` not found (required by /home/roboterax/carbgym/python/isaacgym/_bindings/linux64/gym_36.so)

# solution
mkdir ${YOUR_CONDA_ENV}/lib/_unused
mv ${YOUR_CONDA_ENV}/lib/libstdc++* ${YOUR_CONDA_ENV}/lib/_unused
```


## 安装

1. 使用 Python 3.8 创建新的 conda 虚拟环境：`conda create -n myenv python=3.8`。
2. 为获得最佳性能，建议使用 NVIDIA 驱动版本 525：`sudo apt install nvidia-driver-525`。最低支持版本为 515。如果无法安装 525，确保系统至少为 515 以维持基本功能。
3. 安装 PyTorch 1.13 并使用 Cuda-11.7：
   - `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
4. 安装 numpy-1.23：`conda install numpy=1.23`。
5. 安装 Isaac Gym：
   - 从 https://developer.nvidia.com/isaac-gym 下载并安装 Isaac Gym Preview 4。
   - `cd isaacgym/python && pip install -e .`
   - 运行示例：`cd examples && python 1080_balls_of_solitude.py`。
   - 如遇问题，请查阅 `isaacgym/docs/index.html`。
6. 安装 huma_robot_gym：
   - 克隆本仓库。
   - `cd huma_robot_gym && pip install -e .`


## 使用指南

#### 示例

```bash
# 在 humanoid 目录下
python scripts/train.py --task=t1pro_s_ppo--headless --num_envs 4096

# 评估训练好的 PPO 策略
python scripts/play.py  --task=t1pro_s_ppo --num_envs 1 

# 实现仿真到仿真（sim-to-sim）模型转换
python scripts/fdsim2ros.py 
```

#### 1. 默认任务

- **t1pro_s_ppo**
   - 目的：基线，PPO 策略，多帧低级控制

#### 2. PPO 策略
- **训练命令**：训练 PPO 策略请运行：
  ```
  python humanoid/scripts/train.py --task=t1pro_s_ppo --headless --num_envs 4096
  ```
- **运行已训练策略**：部署已训练的 PPO 策略请使用：
  ```
  python humanoid/scripts/play.py --task=t1pro_s_ppo 
  ```
- 默认会加载上次运行实验文件夹中最新的模型。也可以通过在训练配置中调整 `load_run` 和 `checkpoint` 来选择其他运行/模型。

#### 3. Sim-to-sim
- **注意：在开始 sim-to-sim 之前，请先运行 `play.py` 导出 JIT 策略。**
- **基于 Mujoco 的 Sim2Sim 部署**：使用如下命令执行 sim2sim：
  ```
  python scripts/fdsim2ros.py
  ```


#### 4. 参数
- **CPU 与 GPU 使用**：如需在 CPU 上运行仿真，请同时设置 `--sim_device=cpu` 和 `--rl_device=cpu`。如需使用 GPU，请指定 `--sim_device=cuda:{0,1,2...}` 和 `--rl_device={0,1,2...}`。注意 `CUDA_VISIBLE_DEVICES` 不适用，且需匹配 `--sim_device` 与 `--rl_device` 设置。
- **无头模式**：加入 `--headless` 可在不渲染的情况下运行。
- **渲染控制**：训练过程中按 'v' 切换渲染。
- **策略保存位置**：训练得到的策略保存在 `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。

## 代码结构

1. 每个环境依赖一个 `env` 文件（如 `legged_robot.py`）和一个 `configuration` 文件（如 `legged_robot_config.py`）。后者包含两个类：`LeggedRobotCfg`（包含所有环境参数）和 `LeggedRobotCfgPPO`（包含所有训练参数）。
2. `env` 和 `config` 类均采用继承机制。
3. 在 `cfg` 中非零的奖励权重会将同名的函数加入总奖励中。
4. 任务需使用 `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` 注册。注册可在 `envs/__init__.py` 内，也可在仓库外部进行。


## 添加新环境

基础环境 `legged_robot` 构建了一个粗糙地形的 locomotion 任务。对应配置默认不指定机器人资源（URDF/MJCF）且无奖励权重。

1. 若需添加新环境，在 `envs/` 目录中新建文件夹，并创建名为 `<your_env>_config.py` 的配置文件。新配置应继承已有环境的配置。
2. 若添加新机器人：
    - 将对应资产放入 `resources/` 文件夹。
    - 在 `cfg` 文件中设置资产路径、躯体名称、默认关节位置和 PD 增益。指定所需的 `train_cfg` 和环境的类名（python 类）。
    - 在 `train_cfg` 中设置 `experiment_name` 和 `run_name`。
3. 如有需要，在 `<your_env>.py` 中创建你的环境。继承现有环境，重写所需函数和/或添加奖励函数。
4. 在 `humanoid/envs/__init__.py` 中注册你的环境。
5. 根据需要在 `cfg` 或 `cfg_train` 中修改或调优参数。若要移除某个奖励，将其权重设为零。请避免修改其他环境的参数！
6. 若希望新机器人/环境支持 sim2sim，可能需要修改 `humanoid/scripts/fdsim2ros.py`：
    - 检查机器人在 MJCF 与 URDF 之间的关节映射。
    - 根据训练策略调整机器人的初始关节位置。

## 故障排查

请参考以下常见情况：

```bash
# 错误
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

# 解决
# 设置正确的路径
export LD_LIBRARY_PATH="~/miniconda3/envs/your_env/lib:$LD_LIBRARY_PATH" 

# 或者
sudo apt install libpython3.8

# 错误
AttributeError: module 'distutils' has no attribute 'version'

# 解决
# 安装 pytorch 1.12.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 错误：因 conda 自带的 libstdc++ 版本与系统用于构建 Isaac Gym 的版本不同导致
ImportError: /home/roboterax/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20` not found (required by /home/roboterax/carbgym/python/isaacgym/_bindings/linux64/gym_36.so)

# 解决
mkdir ${YOUR_CONDA_ENV}/lib/_unused
mv ${YOUR_CONDA_ENV}/lib/libstdc++* ${YOUR_CONDA_ENV}/lib/_unused