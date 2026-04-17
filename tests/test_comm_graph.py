"""Tests for red_within_blue.comm_graph (pure-JAX communication graph utilities)."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.comm_graph import (
    build_adjacency,
    compute_components,
    compute_degree,
    compute_isolated,
    get_agent_isolation_duration,
    get_fragmentation_count,
    init_tracker,
    route_messages,
    update_tracker,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_f32 = jnp.float32
_i32 = jnp.int32


def _positions(*pts):
    """Convenience: build [N, 2] float32 from (x, y) tuples."""
    return jnp.array(pts, dtype=_f32)


def _ranges(*vals):
    """Convenience: build [N] float32 comm ranges."""
    return jnp.array(vals, dtype=_f32)


# ── 15. test_adjacency_within_range ─────────────────────────────────────────

def test_adjacency_within_range():
    """Agents within each other's comm_radius should be connected."""
    # Two agents 3.0 apart; both have range 5.0
    pos = _positions((0.0, 0.0), (3.0, 0.0))
    rng = _ranges(5.0, 5.0)
    adj = build_adjacency(pos, rng)
    assert adj[0, 1] == True  # noqa: E712
    assert adj[1, 0] == True  # noqa: E712


# ── 16. test_adjacency_out_of_range ─────────────────────────────────────────

def test_adjacency_out_of_range():
    """Agents beyond comm_radius should NOT be connected."""
    pos = _positions((0.0, 0.0), (10.0, 0.0))
    rng = _ranges(5.0, 5.0)
    adj = build_adjacency(pos, rng)
    assert adj[0, 1] == False  # noqa: E712
    assert adj[1, 0] == False  # noqa: E712


# ── 17. test_adjacency_no_self_loop ─────────────────────────────────────────

def test_adjacency_no_self_loop():
    """Diagonal of the adjacency matrix must always be False."""
    pos = _positions((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    rng = _ranges(100.0, 100.0, 100.0)
    adj = build_adjacency(pos, rng)
    for i in range(3):
        assert adj[i, i] == False  # noqa: E712


# ── 18. test_adjacency_asymmetric ───────────────────────────────────────────

def test_adjacency_asymmetric():
    """Different comm_ranges can produce an asymmetric adjacency matrix."""
    # Agent 0 has range 5, agent 1 has range 2. Distance = 3.
    pos = _positions((0.0, 0.0), (3.0, 0.0))
    rng = _ranges(5.0, 2.0)
    adj = build_adjacency(pos, rng)
    assert adj[0, 1] == True   # 0 can reach 1 (dist 3 <= 5)  # noqa: E712
    assert adj[1, 0] == False  # 1 cannot reach 0 (dist 3 > 2)  # noqa: E712


# ── 19. test_compute_components_connected ───────────────────────────────────

def test_compute_components_connected():
    """All agents within range -> single connected component."""
    pos = _positions((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    rng = _ranges(5.0, 5.0, 5.0)
    adj = build_adjacency(pos, rng)
    num_c, is_conn = compute_components(adj)
    assert int(num_c) == 1
    assert bool(is_conn) is True


# ── 20. test_compute_components_fragmented ──────────────────────────────────

def test_compute_components_fragmented():
    """Two isolated clusters -> num_components == 2."""
    # Cluster A: (0,0),(1,0)  Cluster B: (100,0),(101,0)  range 5
    pos = _positions((0.0, 0.0), (1.0, 0.0), (100.0, 0.0), (101.0, 0.0))
    rng = _ranges(5.0, 5.0, 5.0, 5.0)
    adj = build_adjacency(pos, rng)
    num_c, is_conn = compute_components(adj)
    assert int(num_c) == 2
    assert bool(is_conn) is False


# ── 21. test_compute_isolated ───────────────────────────────────────────────

def test_compute_isolated():
    """Agent far from everyone else should be flagged isolated."""
    pos = _positions((0.0, 0.0), (1.0, 0.0), (100.0, 0.0))
    rng = _ranges(5.0, 5.0, 5.0)
    adj = build_adjacency(pos, rng)
    deg = compute_degree(adj)
    iso = compute_isolated(deg)
    # Agent 2 is 100 units away — isolated
    assert bool(iso[2]) is True
    # Agents 0 and 1 are close — not isolated
    assert bool(iso[0]) is False
    assert bool(iso[1]) is False


# ── 22. test_route_messages_mean ────────────────────────────────────────────

def test_route_messages_mean():
    """Verify mean aggregation with known message vectors."""
    # 3 agents all connected (range large enough)
    pos = _positions((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    rng = _ranges(10.0, 10.0, 10.0)
    adj = build_adjacency(pos, rng)

    msgs = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=_f32)

    routed = route_messages(adj, msgs)

    # Agent 0 receives from agents 1 and 2 -> mean([0,1],[1,1]) = [0.5, 1.0]
    assert jnp.allclose(routed[0], jnp.array([0.5, 1.0]), atol=1e-5)
    # Agent 1 receives from agents 0 and 2 -> mean([1,0],[1,1]) = [1.0, 0.5]
    assert jnp.allclose(routed[1], jnp.array([1.0, 0.5]), atol=1e-5)
    # Agent 2 receives from agents 0 and 1 -> mean([1,0],[0,1]) = [0.5, 0.5]
    assert jnp.allclose(routed[2], jnp.array([0.5, 0.5]), atol=1e-5)


# ── 23. test_route_messages_isolated ────────────────────────────────────────

def test_route_messages_isolated():
    """Isolated agent (no incoming) should receive a zero vector."""
    pos = _positions((0.0, 0.0), (100.0, 0.0))
    rng = _ranges(5.0, 5.0)
    adj = build_adjacency(pos, rng)

    msgs = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=_f32)
    routed = route_messages(adj, msgs)

    assert jnp.allclose(routed[0], jnp.zeros(2))
    assert jnp.allclose(routed[1], jnp.zeros(2))


# ── 24. test_tracker_timeline_write ─────────────────────────────────────────

def test_tracker_timeline_write():
    """GraphTracker records data at the correct step index."""
    num_agents = 2
    node_dim = 3
    tracker = init_tracker(max_steps=5, num_agents=num_agents, node_feature_dim=node_dim)

    adj = jnp.array([[False, True], [True, False]])
    deg = jnp.array([1, 1], dtype=_i32)
    num_c = jnp.int32(1)
    is_conn = jnp.bool_(True)
    iso = jnp.array([False, False])
    nf = jnp.ones((num_agents, node_dim), dtype=_f32)

    tracker = update_tracker(tracker, adj, deg, num_c, is_conn, iso, nf)

    # Written at index 0
    assert int(tracker.current_step) == 1
    assert bool(tracker.is_connected_timeline[0]) is True
    assert int(tracker.degree_timeline[0, 0]) == 1
    assert jnp.allclose(tracker.node_features[0], nf)

    # Index 1 should still be zeros
    assert bool(tracker.is_connected_timeline[1]) is False
    assert int(tracker.degree_timeline[1, 0]) == 0


# ── 25. test_tracker_fragmentation_count ────────────────────────────────────

def test_tracker_fragmentation_count():
    """Fragmentation counter counts disconnected steps correctly."""
    num_agents = 3
    node_dim = 2
    tracker = init_tracker(max_steps=5, num_agents=num_agents, node_feature_dim=node_dim)

    nf = jnp.zeros((num_agents, node_dim), dtype=_f32)

    # Step 0: connected
    adj0 = jnp.ones((3, 3), dtype=jnp.bool_) & ~jnp.eye(3, dtype=jnp.bool_)
    deg0 = compute_degree(adj0)
    tracker = update_tracker(
        tracker, adj0, deg0, jnp.int32(1), jnp.bool_(True),
        compute_isolated(deg0), nf,
    )

    # Step 1: fragmented
    adj1 = jnp.zeros((3, 3), dtype=jnp.bool_)
    deg1 = compute_degree(adj1)
    tracker = update_tracker(
        tracker, adj1, deg1, jnp.int32(3), jnp.bool_(False),
        compute_isolated(deg1), nf,
    )

    # Step 2: fragmented again
    tracker = update_tracker(
        tracker, adj1, deg1, jnp.int32(3), jnp.bool_(False),
        compute_isolated(deg1), nf,
    )

    frag = get_fragmentation_count(tracker)
    assert int(frag) == 2  # steps 1 and 2 disconnected


# ── 26. test_tracker_isolation_duration ─────────────────────────────────────

def test_tracker_isolation_duration():
    """Per-agent isolation duration should count correctly."""
    num_agents = 2
    node_dim = 2
    tracker = init_tracker(max_steps=4, num_agents=num_agents, node_feature_dim=node_dim)
    nf = jnp.zeros((num_agents, node_dim), dtype=_f32)

    # Step 0: both connected
    adj_conn = jnp.array([[False, True], [True, False]])
    deg_conn = compute_degree(adj_conn)
    iso_conn = compute_isolated(deg_conn)
    tracker = update_tracker(
        tracker, adj_conn, deg_conn, jnp.int32(1), jnp.bool_(True),
        iso_conn, nf,
    )

    # Step 1: agent 1 isolated (adj all False)
    adj_iso = jnp.array([[False, False], [False, False]])
    deg_iso = compute_degree(adj_iso)
    iso_iso = compute_isolated(deg_iso)
    tracker = update_tracker(
        tracker, adj_iso, deg_iso, jnp.int32(2), jnp.bool_(False),
        iso_iso, nf,
    )

    # Step 2: agent 1 isolated again
    tracker = update_tracker(
        tracker, adj_iso, deg_iso, jnp.int32(2), jnp.bool_(False),
        iso_iso, nf,
    )

    # Agent 0: isolated at steps 1, 2 -> duration 2
    assert int(get_agent_isolation_duration(tracker, 0)) == 2
    # Agent 1: isolated at steps 1, 2 -> duration 2
    assert int(get_agent_isolation_duration(tracker, 1)) == 2
