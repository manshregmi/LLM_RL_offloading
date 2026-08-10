"""
Whole-graph latency-optimal edge/cloud partition solver.

The public API is the same as the previous brute-force solver:
    evaluate_assignment(...)
    run_dijkstra_partition(...)

0 = Edge
1 = Cloud
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from profiling.profiling_class import ProfilingData


def _evaluate_layer(
    profiling: ProfilingData,
    layer_idx: int,
    action_pattern: Tuple[int, ...],
    assignment_prefix: Tuple[int, ...],
    flat_index: Dict[Tuple[int, int], int],
    bw: float,
    rtt: float,
    contention_map: Dict[Tuple[int, int], float],
    cloud_pending_ms: float,
) -> Tuple[float, float]:
    """Return latency for one layer and cloud waiting time for the next layer."""

    layer = profiling.layers[layer_idx]
    num_nodes = len(layer)

    action = np.zeros((num_nodes, 2), dtype=int)
    action[:, 0] = layer_idx
    action[:, 1] = action_pattern

    # ---------------- Communication ----------------
    transmission_times = []

    if layer_idx > 0:
        for node_idx, node_id in enumerate(layer):
            curr_loc = action[node_idx, 1]
            parent_nodes = profiling.dependencies.get((layer_idx, node_id), [])

            for parent_layer, parent_node in parent_nodes:
                parent_key = (parent_layer, parent_node)

                if parent_key not in flat_index:
                    raise ValueError(
                        f"Dependency {parent_key} does not exist in profiling.layers"
                    )

                parent_flat_idx = flat_index[parent_key]

                if parent_flat_idx >= len(assignment_prefix):
                    raise ValueError(
                        f"Invalid dependency {parent_key} -> "
                        f"({layer_idx}, {node_id}). Parent must be in an earlier layer."
                    )

                parent_loc = assignment_prefix[parent_flat_idx]

                if parent_loc != curr_loc:
                    output_size_mb = profiling.get_output_size(layer_idx, node_id)
                    tx_time = (output_size_mb) / (bw)
                    transmission_times.append(max(tx_time, rtt / 1000.0))

    else:
        # First layer: upload input for nodes assigned to cloud.
        for node_idx in range(num_nodes):
            if action[node_idx, 1] == 1:
                input_size_mb = profiling.get_input_size()
                tx_time = (input_size_mb) / (bw)
                transmission_times.append(max(tx_time, rtt / 1000.0))

    max_transmission_time = max(transmission_times) if transmission_times else 0.0

    # ---------------- Edge computation ----------------
    edge_times_ms = [
        profiling.get_node_edge_time(layer_idx, node_id)
        for node_idx, node_id in enumerate(layer)
        if action[node_idx, 1] == 0
    ]

    # Keep the same behavior as simulator.py: edge work in a layer is summed.
    edge_time_ms = sum(edge_times_ms)
    edge_time_s = edge_time_ms / 1000.0

    # ---------------- Cloud waiting ----------------
    cloud_nodes = np.where(action[:, 1] == 1)[0]

    actual_idle_time_s = 0.0
    if len(cloud_nodes) > 0:
        actual_idle_time_s = max(
            0.0,
            cloud_pending_ms / 1000.0 - edge_time_s,
        )

    layer_latency_s = (
        edge_time_s
        + max_transmission_time
        + actual_idle_time_s
    )

    # ---------------- Cloud work for next layer ----------------
    if len(cloud_nodes) == 0:
        return layer_latency_s, 0.0
    effective_cloud_times_ms = []

    for node_idx in cloud_nodes:

        node_id = layer[node_idx]

        cloud_proc_ms = profiling.get_node_cloud_time(
            layer_idx,
            node_id
        )

        contention_extra_ms = contention_map.get(
            (layer_idx, node_idx),
            0.0
        )

    effective_cloud_times_ms.append(
        max(0.0, cloud_proc_ms)
        + max(0.0, contention_extra_ms)
    )

    cloud_time_ms = max(effective_cloud_times_ms)

    # cloud_proc_ms = max(
    #     profiling.get_node_cloud_time(layer_idx, layer[node_idx])
    #     for node_idx in cloud_nodes
    # )

    # # contention_generator.get_contention_extra_map() stores one value per node.
    # contention_extra = max(
    #     contention_map.get((layer_idx, node_idx), 0.0)
    #     for node_idx in cloud_nodes
    # )

    new_cloud_pending_ms = max(
        0.0,
        cloud_pending_ms - edge_time_ms,
    )
    new_cloud_pending_ms += cloud_proc_ms 

    return layer_latency_s, new_cloud_pending_ms


def evaluate_assignment(
    profiling: ProfilingData,
    bw: float,                      # Mbps
    rtt: float,                     # ms
    contention_map: Dict[Tuple[int, int], float],
    assignment: List[int],
) -> float:
    """Compute total latency for one complete whole-graph assignment."""

    total_nodes = sum(len(layer) for layer in profiling.layers)

    if len(assignment) != total_nodes:
        print(f"Assignment length is {len(assignment)}, expected {total_nodes}")

    flat_index = {}
    flat_idx = 0

    for layer_idx, layer in enumerate(profiling.layers):
        for node_id in layer:
            flat_index[(layer_idx, node_id)] = flat_idx
            flat_idx += 1

    total_latency = 0.0
    cloud_pending_ms = 0.0
    flat_idx = 0

    for layer_idx, layer in enumerate(profiling.layers):
        num_nodes = len(layer)
        action_pattern = tuple(
            assignment[flat_idx + node_idx]
            for node_idx in range(num_nodes)
        )

        layer_latency, cloud_pending_ms = _evaluate_layer(
            profiling,
            layer_idx,
            action_pattern,
            tuple(assignment[:flat_idx]),
            flat_index,
            bw,
            rtt,
            contention_map,
            cloud_pending_ms,
        )

        total_latency += layer_latency
        flat_idx += num_nodes

    return total_latency


def run_dijkstra_partition(
    profiling_data: ProfilingData,
    bw_pred: float,
    rtt_pred: float,
    contention_extra_map: Dict[Tuple[int, int], float],
) -> Tuple[List[int], float]:
    """
    Find the globally minimum-latency whole-graph assignment.

    This replaces exhaustive 2^N enumeration with dynamic programming.
    At each layer, every valid edge/cloud action is explored, but a state is
    removed when another state with the same future dependency information has
    both lower latency and lower cloud waiting time.
    """

    layers = profiling_data.layers

    if not layers:
        return [], 0.0

    if bw_pred <= 0:
        raise ValueError("Bandwidth must be greater than zero")

    # Flattened location of every graph node.
    flat_index = {}
    flat_idx = 0

    for layer_idx, layer in enumerate(layers):
        for node_id in layer:
            flat_index[(layer_idx, node_id)] = flat_idx
            flat_idx += 1

    # Remember a past node only until its final future dependency is processed.
    last_dependency_use = {}

    for (child_layer, _), parent_nodes in profiling_data.dependencies.items():
        for parent_layer, parent_node in parent_nodes:
            parent_key = (parent_layer, parent_node)
            last_dependency_use[parent_key] = max(
                child_layer,
                last_dependency_use.get(parent_key, -1),
            )

    # State: (total_latency_s, cloud_pending_ms, flattened_assignment)
    states = [(0.0, 0.0, ())]

    for layer_idx, layer in enumerate(layers):
        num_nodes = len(layer)

        # Same rule as CloudEdgeSimulator: final layer executes on Edge.
        if layer_idx == len(layers) - 1:
            possible_actions = [tuple(0 for _ in range(num_nodes))]
        else:
            possible_actions = [
                tuple(
                    (mask >> node_idx) & 1
                    for node_idx in range(num_nodes)
                )
                for mask in range(1 << num_nodes)
            ]

        # Past assignments that can still affect a future transmission.
        active_dependencies = sorted(
            node_key
            for node_key, last_use in last_dependency_use.items()
            if node_key[0] <= layer_idx and last_use > layer_idx
        )

        # dependency_signature -> Pareto-optimal states
        next_states = {}

        for total_latency, cloud_pending_ms, assignment in states:
            for action_pattern in possible_actions:
                layer_latency, new_cloud_pending_ms = _evaluate_layer(
                    profiling_data,
                    layer_idx,
                    action_pattern,
                    assignment,
                    flat_index,
                    bw_pred,
                    rtt_pred,
                    contention_extra_map,
                    cloud_pending_ms,
                )

                new_assignment = assignment + action_pattern
                new_total_latency = total_latency + layer_latency

                dependency_signature = tuple(
                    new_assignment[flat_index[node_key]]
                    for node_key in active_dependencies
                )

                candidate = (
                    new_total_latency,
                    new_cloud_pending_ms,
                    new_assignment,
                )

                frontier = next_states.setdefault(dependency_signature, [])

                # Existing state dominates this candidate.
                if any(
                    old_latency <= new_total_latency
                    and old_pending <= new_cloud_pending_ms
                    for old_latency, old_pending, _ in frontier
                ):
                    continue

                # Remove states dominated by this candidate.
                frontier[:] = [
                    old_state
                    for old_state in frontier
                    if not (
                        new_total_latency <= old_state[0]
                        and new_cloud_pending_ms <= old_state[1]
                    )
                ]

                frontier.append(candidate)

        states = [
            state
            for frontier in next_states.values()
            for state in frontier
        ]

        if not states:
            raise RuntimeError(
                f"No valid partition state remained after layer {layer_idx}"
            )

    best_latency, _, best_assignment = min(
        states,
        key=lambda state: state[0],
    )

    return list(best_assignment), best_latency