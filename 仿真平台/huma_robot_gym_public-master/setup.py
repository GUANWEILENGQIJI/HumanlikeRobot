# Copyright (c) 2024-2024, fudeRobot All rights reserved.

from setuptools import find_packages
from distutils.core import setup

setup(
    name='humanoid',
    version='1.0.0',
    author='Zhongyuan Deng',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='dzy810488037@gmail.com',
    description='Isaac Gym environments for humanoid robot',
    install_requires=['isaacgym', 
                      'wandb',
                      'tensorboard',
                      'tqdm',
                      'numpy==1.23.5',
                      'opencv-python',
                      'mujoco',
                      'matplotlib']
)
