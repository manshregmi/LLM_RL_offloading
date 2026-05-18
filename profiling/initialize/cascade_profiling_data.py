from profiling.profiling_class import ProfilingData

def cascade_profiling(profiling_data: ProfilingData, n: int) -> ProfilingData:
    """
    Create a new ProfilingData where the original DAG is repeated n times back-to-back.
    The last node(s) of copy i become predecessors of the first node(s) of copy i+1.
    """
    original = profiling_data
    L = len(original.layers)                     # number of layers in one copy
    sources = [(0, node) for node in original.layers[0]]   # original source nodes
    sinks = [(L-1, node) for node in original.layers[-1]]  # original sink nodes

    # 1. Build layers list: just repeat the original layer structure
    new_layers = original.layers * n

    # 2. Prepare dictionaries
    new_node_edge_times = {}
    new_node_cloud_times = {}
    new_input_size = {}
    new_dependencies = {}

    # Helper: shift key (layer, node) by offset
    def shift_key(layer, node, offset):
        return (layer + offset, node)

    # 3. Copy and shift all node data for each copy
    for copy in range(n):
        offset = copy * L
        for (l, node), value in original.node_edge_times.items():
            new_node_edge_times[shift_key(l, node, offset)] = value
        for (l, node), value in original.node_cloud_times.items():
            new_node_cloud_times[shift_key(l, node, offset)] = value
        for (l, node), value in original.input_size.items():
            new_input_size[shift_key(l, node, offset)] = value

    # 4. Copy and shift dependencies (internal edges only)
    for copy in range(n):
        offset = copy * L
        for (l, node), preds in original.dependencies.items():
            new_key = shift_key(l, node, offset)
            new_preds = [shift_key(pred_l, pred_node, offset) for (pred_l, pred_node) in preds]
            new_dependencies[new_key] = new_preds

    # 5. Add cross‑copy dependencies and adjust input sizes
    # For each pair copy i -> copy i+1
    for copy in range(n - 1):
        offset_this = copy * L
        offset_next = (copy + 1) * L

        # Determine output size of this copy's sinks (all sinks have same output size)
        # Use the first sink's output size from original
        sink_node = sinks[0]                     # (L-1, node)
        output_size = original.get_output_size(sink_node[0], sink_node[1])  # from original

        # For each source in the next copy, add dependency on each sink of this copy
        for src_l, src_node in sources:
            next_source = (offset_next + src_l, src_node)
            # Initialize dependency list if not present (should be present from copy)
            if next_source not in new_dependencies:
                new_dependencies[next_source] = []
            # Add all sinks of this copy as predecessors
            for sink_l, sink_node in sinks:
                this_sink = (offset_this + sink_l, sink_node)
                new_dependencies[next_source].append(this_sink)

            # Override input size for this source to be the sink's output size
            new_input_size[next_source] = output_size

    # 6. Preserve other parameters unchanged
    return ProfilingData(
        numberOfEdgeDevice=original.numberOfEdgeDevice,
        layers=new_layers,
        node_edge_times=new_node_edge_times,
        node_cloud_times=new_node_cloud_times,
        bandwidth=original.bandwidth,
        rtt=original.rtt,
        input_size=new_input_size,
        node_edge_powers={},           # ignore powers as requested
        edge_idle_power=original.edge_idle_power,
        deadline=original.deadline,
        edge_communication_power=original.edge_communication_power,
        dependencies=new_dependencies,
    )