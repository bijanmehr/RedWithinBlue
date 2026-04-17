"""Training infrastructure for RedWithinBlue experiments."""

from red_within_blue.training.config import (
    EnvParams,
    NetworkParams,
    TrainParams,
    RewardParams,
    ExperimentConfig,
    get_stage_configs,
)

from red_within_blue.training.networks import Actor, Critic, QNetwork
from red_within_blue.training.rewards_training import (
    normalized_exploration_reward,
    multi_agent_reward,
    make_multi_agent_reward,
)
from red_within_blue.training.rollout import (
    Trajectory, MultiTrajectory,
    collect_episode_scan, collect_episode_multi_scan,
)
from red_within_blue.training.losses import (
    compute_discounted_returns,
    pg_loss,
    pg_loss_with_baseline,
    actor_critic_loss,
)
from red_within_blue.training.checkpoint import (
    flatten_params,
    unflatten_params,
    save_checkpoint,
    load_checkpoint,
)
from red_within_blue.training.trainer import make_train, make_train_multi_seed
from red_within_blue.training.dqn import (
    tabular_q_update,
    dqn_loss,
    compute_dqn_targets,
    epsilon_greedy,
)
from red_within_blue.training.metrics import (
    compute_coverage,
    compute_action_distribution,
    compute_explained_variance,
    compute_steps_to_coverage,
    compute_connectivity_fraction,
)
from red_within_blue.training.stats import (
    chi_squared_vs_uniform,
    welch_t_test,
    bonferroni_correct,
)
from red_within_blue.training.transfer import (
    transfer_actor_params,
    init_fresh_critic,
    compute_cka,
    extract_hidden_features,
)
from red_within_blue.training.plotting import apply_style, plot_stage_summary, COLORS
from red_within_blue.training.gif import record_episode_gif
from red_within_blue.training.report import generate_report
