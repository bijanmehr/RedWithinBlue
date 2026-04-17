"""Communication-graph utilities for multi-agent RL (pure JAX, JIT-compatible)."""

import jax
import jax.numpy as jnp

from red_within_blue.types import GraphTracker

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_adjacency(positions: jax.Array, comm_ranges: jax.Array) -> jax.Array:
    """Build a directed adjacency matrix from agent positions and comm ranges.

    Parameters
    ----------
    positions : Array[N, 2]
        Agent positions (float32 or int32).
    comm_ranges : Array[N]
        Per-agent communication range (float32).  Agent *i* can **send** to
        agent *j* when ``||pos[i] - pos[j]|| <= comm_ranges[i]``.

    Returns
    -------
    Array[N, N] bool
        ``adj[i, j] = True`` means agent *i* can send to *j*.
        Diagonal is always False (no self-loops).
    """
    pos = positions.astype(jnp.float32)
    diff = pos[:, None, :] - pos[None, :, :]          # [N, N, 2]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))      # [N, N]
    in_range = dist <= comm_ranges[:, None]             # [N, N]
    no_self = ~jnp.eye(positions.shape[0], dtype=jnp.bool_)
    return in_range & no_self


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------


def route_messages(adjacency: jax.Array, messages_out: jax.Array) -> jax.Array:
    """Mean-pool incoming messages along the communication graph.

    Parameters
    ----------
    adjacency : Array[N, N] bool
        ``adjacency[j, i]`` is True when agent *j* can send to agent *i*.
    messages_out : Array[N, D] float32
        Outgoing message vectors.

    Returns
    -------
    Array[N, D] float32
        ``messages_in[i]`` = mean of ``messages_out[j]`` over all *j* where
        ``adjacency[j, i]``.  Zeros if agent *i* has no incoming neighbours.
    """
    adj_f = adjacency.astype(jnp.float32)              # [N, N]
    incoming = adj_f.T @ messages_out                   # [N, D]
    degree = adj_f.T.sum(axis=1)                        # [N]
    return incoming / jnp.maximum(degree[:, None], 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_degree(adjacency: jax.Array) -> jax.Array:
    """Outgoing degree (row sum) for each agent.

    Returns Array[N] int32.
    """
    return adjacency.astype(jnp.int32).sum(axis=1)


def compute_components(adjacency: jax.Array):
    """Spectral connected-component analysis on the symmetrised graph.

    Returns
    -------
    num_components : scalar int32
    is_connected   : scalar bool
    """
    sym = adjacency | adjacency.T
    adj_f = sym.astype(jnp.float32)
    deg_vec = adj_f.sum(axis=1)                         # [N]
    laplacian = jnp.diag(deg_vec) - adj_f               # [N, N]
    eigenvalues = jnp.linalg.eigvalsh(laplacian)        # [N]
    num_components = jnp.sum(eigenvalues < 1e-4).astype(jnp.int32)
    is_connected = num_components == 1
    return num_components, is_connected


def compute_isolated(degree: jax.Array) -> jax.Array:
    """Boolean mask: True where an agent has zero outgoing degree."""
    return degree == 0


# ---------------------------------------------------------------------------
# GraphTracker management
# ---------------------------------------------------------------------------


def init_tracker(
    max_steps: int,
    num_agents: int,
    node_feature_dim: int,
) -> GraphTracker:
    """Create a zeroed-out :class:`GraphTracker` with preallocated timelines."""
    return GraphTracker(
        adjacency=jnp.zeros((num_agents, num_agents), dtype=jnp.bool_),
        degree=jnp.zeros((num_agents,), dtype=jnp.int32),
        num_components=jnp.int32(0),
        is_connected=jnp.bool_(False),
        adjacency_timeline=jnp.zeros(
            (max_steps, num_agents, num_agents), dtype=jnp.bool_
        ),
        num_components_timeline=jnp.zeros((max_steps,), dtype=jnp.int32),
        is_connected_timeline=jnp.zeros((max_steps,), dtype=jnp.bool_),
        degree_timeline=jnp.zeros((max_steps, num_agents), dtype=jnp.int32),
        isolated_timeline=jnp.zeros((max_steps, num_agents), dtype=jnp.bool_),
        node_features=jnp.zeros(
            (max_steps, num_agents, node_feature_dim), dtype=jnp.float32
        ),
        current_step=jnp.int32(0),
    )


def update_tracker(
    tracker: GraphTracker,
    adjacency: jax.Array,
    degree: jax.Array,
    num_components: jax.Array,
    is_connected: jax.Array,
    isolated: jax.Array,
    node_features: jax.Array,
) -> GraphTracker:
    """Write one timestep of data into the tracker and advance the cursor."""
    idx = tracker.current_step
    return tracker.replace(
        adjacency=adjacency,
        degree=degree,
        num_components=num_components,
        is_connected=is_connected,
        adjacency_timeline=tracker.adjacency_timeline.at[idx].set(adjacency),
        num_components_timeline=tracker.num_components_timeline.at[idx].set(
            num_components
        ),
        is_connected_timeline=tracker.is_connected_timeline.at[idx].set(
            is_connected
        ),
        degree_timeline=tracker.degree_timeline.at[idx].set(degree),
        isolated_timeline=tracker.isolated_timeline.at[idx].set(isolated),
        node_features=tracker.node_features.at[idx].set(node_features),
        current_step=idx + 1,
    )


def get_fragmentation_count(tracker: GraphTracker) -> jax.Array:
    """Number of steps so far where the graph was **not** connected."""
    mask = jnp.arange(tracker.is_connected_timeline.shape[0]) < tracker.current_step
    return jnp.sum(mask & ~tracker.is_connected_timeline).astype(jnp.int32)


def get_agent_isolation_duration(tracker: GraphTracker, agent_idx: int) -> jax.Array:
    """Number of steps where *agent_idx* was isolated (degree == 0)."""
    col = tracker.isolated_timeline[:, agent_idx]
    mask = jnp.arange(col.shape[0]) < tracker.current_step
    return jnp.sum(mask & col).astype(jnp.int32)
