# splitwise_simulation.py
# Run this script after ensuring the necessary modules (profiling_class, simulator) are in your Python path.

import numpy as np
from profiling.profiling_class import ProfilingData
from simulator.simulator import CloudEdgeSimulator   # adjust import if needed (e.g., from your_simulator import CloudEdgeSimulator)


def generate_splitwise_assignment(profiling_data) -> list:
    """
    Create a list of actions (assignment matrices) for all layers
    following the Splitwise policy:
        - Prompt (pre‑fill) layers: all nodes → cloud (1)
        - Token generation layers: all nodes → edge (0)

    The mapping of layers is assumed to be:
        Layer 0: nodes [0: YOLO, 1: Llama‑1 pre‑fill]
        Layers 1..99: node 0 → Llama‑1 token generation
        Layer 100: node 0 → Llama‑2 pre‑fill
        Layers 101..199: node 0 → Llama‑2 token generation
        Layer 300: node 0 → BART pre‑fill
        Layers 301..399: node 0 → BART token generation
    All other layers are assigned to edge (0) by default.
    """
    layers = profiling_data.layers  # list of lists, each inner list holds node indices for that layer
    num_layers = len(layers)

    # Pre‑fill layers: we mark them as cloud (1) for all their nodes
    prefill_layers = {
        0: [1],          # Llama‑1 pre‑fill is node 1 at layer 0 (YOLO node 0 will be handled separately)
        100: [0],        # Llama‑2 pre‑fill at layer 100, node 0
        300: [0]         # BART pre‑fill at layer 300, node 0
    }

    # Optionally assign YOLO (layer 0, node 0) to cloud as part of the initial pipeline
    yolo_to_cloud = True

    splitwise_actions = []

    for layer_idx, nodes in enumerate(layers):
        num_nodes = len(nodes)

        # Default assignment: all nodes go to edge (0)
        assignment = np.zeros((num_nodes, 2), dtype=int)
        assignment[:, 0] = layer_idx

        if layer_idx in prefill_layers:
            # Set specified nodes to cloud
            for node_idx in prefill_layers[layer_idx]:
                if node_idx < num_nodes:
                    assignment[node_idx, 1] = 1
            # For layer 0, also handle YOLO if needed
            if layer_idx == 0 and yolo_to_cloud:
                assignment[0, 1] = 1
        # else: all nodes remain edge (0) → correct for token generation layers

        splitwise_actions.append(assignment)

    return splitwise_actions


def simulate_splitwise(profiling_data, simulator: CloudEdgeSimulator) -> float:
    """
    Simulate the entire DNN using Splitwise assignments.

    Returns:
        Total latency in seconds.
    """
    actions = generate_splitwise_assignment(profiling_data)

    simulator.reset_episode_time()

    # Initial state
    current_bandwidth = simulator.get_current_bandwidth()
    cloud_pending = 0.0
    prev_action = None

    total_latency = 0.0

    for layer_idx in range(len(actions)):
        action = actions[layer_idx]
        current_state = (current_bandwidth, cloud_pending, layer_idx, prev_action)

        # Compute latency for this layer (updates simulator.cumulative_time_seconds)
        layer_latency = simulator.compute_latency(current_state, action, cloud_pending)
        total_latency += layer_latency

        if layer_idx == len(actions) - 1:
            break

        # Prepare for next layer
        is_all_cloud = False
        next_layer = layer_idx + 1
        new_cloud_pending = simulator.get_next_state_cloud_waiting_time(
            next_layer, action, is_all_cloud
        )

        next_state,_ = simulator.get_next_state(current_state, action, new_cloud_pending)
        layer_idx = next_state[2]



        cloud_pending = new_cloud_pending
        current_bandwidth = simulator.get_current_bandwidth()
        prev_action = simulator._action_to_pattern(action)

    return total_latency


if __name__ == "__main__":
    # Load your profiling data (adjust function name if needed)
    from profiling.initialize_agx_profiling import get_LLM_profiling_data  # e.g., profiling_data.py

    profiling_data = get_LLM_profiling_data()

    # Create the simulator (you may need to pass a bandwidth CSV file)
    # If you have a bandwidth file, use: simulator = CloudEdgeSimulator(profiling_data, bandwidth_csv_path="path/to/bw_data.csv")
    # Otherwise, it will fall back to stochastic bandwidth.
    simulator = CloudEdgeSimulator(profiling_data)
    splitwise_total= []
    for x in range(100):
        total = simulate_splitwise(profiling_data, simulator)
        splitwise_total.append(total)
    print(f"Total latency with Splitwise policy: {np.average(splitwise_total)*1000:.4f} seconds")