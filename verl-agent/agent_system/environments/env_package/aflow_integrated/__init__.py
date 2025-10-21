"""
AFlow Integrated Environment for verl-agent
深度集成的 AFlow 环境

This package provides a deeply integrated environment that connects
AFlow's workflow optimization with verl-agent's RL training.
"""

from .envs import build_aflow_envs, AFlowMultiProcessEnv, AFlowWorker

__all__ = ['build_aflow_envs', 'AFlowMultiProcessEnv', 'AFlowWorker']
