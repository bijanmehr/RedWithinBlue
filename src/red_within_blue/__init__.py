"""RedWithinBlue — JAX-native multi-agent grid environment with communication graphs."""

from red_within_blue.types import (
    Action,
    EnvConfig,
    EnvState,
    AgentState,
    GlobalState,
    GridState,
    GraphTracker,
)
from red_within_blue.env import GridCommEnv
from red_within_blue.rewards import (
    exploration_reward,
    revisit_penalty,
    connectivity_reward,
    time_penalty,
    terminal_coverage_bonus,
    competitive_reward,
    compose_rewards,
)
from red_within_blue.wrappers import TrajectoryWrapper
from red_within_blue.replay import ReplayPlayer
from red_within_blue.logger import ExperimentLogger
from red_within_blue.visualizer import render_frame, EnvDashboard

__all__ = [
    # Core env
    "GridCommEnv",
    "EnvConfig",
    "EnvState",
    "Action",
    # State types
    "AgentState",
    "GlobalState",
    "GridState",
    "GraphTracker",
    # Rewards
    "exploration_reward",
    "revisit_penalty",
    "connectivity_reward",
    "time_penalty",
    "terminal_coverage_bonus",
    "competitive_reward",
    "compose_rewards",
    # Wrappers & replay
    "TrajectoryWrapper",
    "ReplayPlayer",
    # Logging
    "ExperimentLogger",
    # Visualization
    "render_frame",
    "EnvDashboard",
]
