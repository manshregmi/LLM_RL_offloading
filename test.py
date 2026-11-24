# Identify the current group
chunk_idx = self._get_chunk_index(layer, num_groups)
cache_key = (num_groups, chunk_idx)

if cache_key in self.group_range_assignments:
    count += 1
    majority_actions = self.group_range_assignments[cache_key]
    print(f"\n[CACHE HIT] Group {chunk_idx}")
    print(f"Cached majority actions: {majority_actions}")

else:
    start_layer, end_layer = self._get_chunk_range(
        chunk_idx, num_groups
    )

    print(f"\n{'='*50}")
    print(f"GROUP {chunk_idx} | K={num_groups}")
    print(f"Levels: {start_layer} to {end_layer}")
    print(f"{'='*50}")

    curr_state = (
        state_bw,
        state_ctime,
        start_layer,
        original_prev_assignment
    )

    votes = {}

    for l in range(start_layer, end_layer + 1):

        action = self._get_policy_action(curr_state)

        # PRINT 1: RL proposals at this level
        print(f"\nLevel {l}:")
        print(f"  RL action: {action[:, 1].tolist()}")
        print(f"  Expected nodes: {self.profiling.get_num_nodes(l)}")
        print(f"  Action rows: {action.shape[0]}")

        for node_idx in range(action.shape[0]):

            if node_idx not in votes:
                votes[node_idx] = []

            votes[node_idx].append(int(action[node_idx, 1]))

        curr_state = (
            state_bw,
            state_ctime,
            l + 1,
            action
        )

    # PRINT 2: Collected votes
    print("\nVotes by node position:")
    for n in sorted(votes):
        print(f"  Node {n}: {votes[n]}")

    # Calculate majority
    majority_actions = np.array([
        np.argmax(np.bincount(votes[n], minlength=2))
        for n in sorted(votes)
    ], dtype=int)

    # PRINT 3: Vote counts and majorities
    print("\nMajority voting results:")
    for n in sorted(votes):
        counts = np.bincount(votes[n], minlength=2)
        print(
            f"  Node {n}: "
            f"Edge votes={counts[0]}, "
            f"Cloud votes={counts[1]}, "
            f"Majority={majority_actions[n]}"
        )

    print(f"\nGroup {chunk_idx} majority vector: {majority_actions}")

    self.group_range_assignments[cache_key] = majority_actions

# Build final action array
current_layer = self.simulator.get_current_layer_index()
num_nodes = self.profiling.get_num_nodes(current_layer)

actions = np.array([
    [current_layer, majority_actions[n]]
    for n in range(num_nodes)
], dtype=int)

# PRINT 4: Final action returned to simulator
print(f"\nFinal action for level {current_layer}:")
print(actions)
print(f"Assignments: {actions[:, 1].tolist()}")