# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2024, fudeRobot All rights reserved.


from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot


from humanoid.utils.task_registry import task_registry


from .t1pro_s.t1pro_config import T1PROCfg, T1PROCfgPPO
from .t1pro_s.t1pro_env import T1PROEnv
task_registry.register( "t1pro_s_ppo", T1PROEnv, T1PROCfg(), T1PROCfgPPO() )

