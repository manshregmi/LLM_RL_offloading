import os
import numpy as np
import pandas as pd
import random
from bisect import bisect_left
from typing import List, Tuple, Optional
from profiling.profiling_class import ProfilingData


# ==================== BANDWIDTH TRACKER ====================

class BandwidthTracker:
    def __init__(self, bandwidth_data: List[Tuple[float, float]]):

        valid_data = [
            (float(t), float(bw))
            for t, bw in bandwidth_data
            if bw is not None and not pd.isna(bw)
        ]

        if not valid_data:
            raise ValueError("No valid bandwidth data provided")

        valid_data.sort(key=lambda x: x[0])

        self.timestamps = np.array([t for t, _ in valid_data], dtype=float)
        self.bandwidths = np.array([bw for _, bw in valid_data], dtype=float)

        self.min_timestamp = float(self.timestamps[0])
        self.normalized_timestamps = self.timestamps - self.min_timestamp

        print(f"✅ Loaded {len(self.timestamps)} bandwidth samples")
        print(f"   Duration: {self.normalized_timestamps[-1]:.2f}s")
        print(f"   BW Range: {self.bandwidths.min():.1f}-{self.bandwidths.max():.1f} Mbps")

    def get_bandwidth_at_time(self, time_seconds: float, use_normalized: bool = False) -> float:

        if use_normalized:
            query_time = float(time_seconds)
        else:
            query_time = float(time_seconds - self.min_timestamp)

        if query_time <= self.normalized_timestamps[0]:
            return float(self.bandwidths[0])

        if query_time >= self.normalized_timestamps[-1]:
            return float(self.bandwidths[-1])

        idx = bisect_left(self.normalized_timestamps, query_time)

        t0 = self.normalized_timestamps[idx - 1]
        t1 = self.normalized_timestamps[idx]
        b0 = self.bandwidths[idx - 1]
        b1 = self.bandwidths[idx]

        ratio = (query_time - t0) / (t1 - t0)
        bandwidth = b0 + ratio * (b1 - b0)

        return float(max(1.0, bandwidth))  # Mbps


# ==================== CSV LOADER ====================

def load_bandwidth_data_from_csv(csv_path: str) -> List[Tuple[float, float]]:

    df = pd.read_csv(csv_path)

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['bandwidth_mbps'] = pd.to_numeric(df['bandwidth_mbps'], errors='coerce')

    df = df.dropna(subset=['timestamp', 'bandwidth_mbps'])

    data = list(zip(
        df['timestamp'].astype(float).values,
        df['bandwidth_mbps'].astype(float).values
    ))

    print(f"📊 Loaded {len(data)} valid bandwidth samples")

    return data


def get_contention_data(contention_csv_path, n_yolos_inference, n_llama_inference, n_bart_inference):
    df = pd.read_csv(contention_csv_path)

    row = df[
        (df['n_yolos'] == n_yolos_inference) &
        (df['n_llama'] == n_llama_inference) &
        (df['n_bart'] == n_bart_inference)
    ]

    if row.empty:
        return {
            "Llama contention ": 0.0,
            "Yolos contention": 0.0,
            "Bart contention": 0.0
        }

    return {
        "Llama contention ": float(row.iloc[0]['Llama contention']),
        "Yolos contention": float(row.iloc[0]['Yolos contention']),
        "Bart contention": float(row.iloc[0]['Bart contention'])
    }

# ==================== CLOUD EDGE SIMULATOR - PURE LATENCY ====================

class CloudEdgeSimulator:
    """
    Pure Latency Minimization Simulator
    - No deadlines, no surplus, no negative count
    - Only goal: minimize total inference latency
    """
    
    def __init__(self, profiling_data: ProfilingData, bandwidth_csv_path: Optional[str] = None, total_pipeline=1):

        self.profiling = profiling_data
        self.bandwidth_tracker = None
        self.cumulative_time_seconds = 0.0
        self.episode_offset = 0.0
        self.episode_start_time = 0.0
        self.i = random.randint(0, 3)
        self.j = random.randint(0, 3)
        self.k = random.randint(0, 3)

        self.total_pipeline = total_pipeline
        # Default bandwidth CSV path


        self.contention_csv_path = os.path.join("simulator", "contention.csv")
        bandwidth_csv_path = os.path.join("simulator", "bw_data.csv")

        if bandwidth_csv_path:
            try:
                bandwidth_data = load_bandwidth_data_from_csv(bandwidth_csv_path)
                if bandwidth_data:
                    self.bandwidth_tracker = BandwidthTracker(bandwidth_data)
            except Exception as e:
                print(f"⚠️ Could not load bandwidth data: {e}")
                print("Falling back to stochastic bandwidth")

    # ================= Episode Time =================

    def reset_episode_time(self):
        """Reset time tracking for a new episode"""
        self.cumulative_time_seconds = 0.0
        self.episode_start_time = 0.0

        if self.bandwidth_tracker:
            max_time = self.bandwidth_tracker.normalized_timestamps[-1]
            self.episode_offset = random.uniform(0, max_time * 0.9)

    def get_current_bandwidth(self) -> float:
        """Get current bandwidth in MBps (Megabytes per second)"""
        if self.bandwidth_tracker:
            query_time = float(self.cumulative_time_seconds + self.episode_offset)
            bw_mbps = self.bandwidth_tracker.get_bandwidth_at_time(
                query_time,
                use_normalized=True
            )
            return float(bw_mbps / 8.0)  # Convert to MBps
        # return float(random.uniform(5, 100))
        # return 100

    # ================= Action Space =================

    def get_possible_actions(self, layer):
        """Generate all possible actions for a given layer"""
        if layer >= len(self.profiling.layers):
            return []

        nodes = self.profiling.get_num_nodes(layer)
        actions = []

        # Last layer: all nodes must be executed on edge
        if layer == len(self.profiling.layers) - 1:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            return [a]

        # Generate all binary patterns for node execution locations
        for pattern in range(2 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1  # 0 = edge, 1 = cloud
            actions.append(a)

        return actions

    # ================= Cloud Waiting Time =================

    def get_next_state_cloud_waiting_time(self, next_layer, current_action, isAllCloud=False):
        """
        Calculate cloud waiting time for the next layer based on current action.
        """
        layer = int(next_layer)
       
        # Find nodes assigned to cloud in current layer
        cloud_nodes = np.where(current_action[:, 1] == 1)[0]

        if (not isAllCloud):
            if (random.random() < 0.2):
                self.i +=1
                self.i = min(self.i, 3)
            elif (random.random() < 0.2):
                self.i -=1
                self.i = max(self.i, 0)
            if (random.random() < 0.2):
                self.j +=1
                self.j = min(self.j, 3)
            elif (random.random() < 0.2):
                self.j -=1
                self.j = max(self.j, 0)
            if (random.random() < 0.2):
                self.k +=1
                self.k = min(self.k, 3)
            elif (random.random() < 0.2):
                self.k -=1
                self.k = max(self.k, 0)


            random_llama_inference = self.i
            random_yolos_inference = self.j
            random_bart_inference = self.k
 
        else:
            random_llama_inference = 3
            random_yolos_inference = 3
            random_bart_inference = 3
 
        contention_row = get_contention_data(
            contention_csv_path=self.contention_csv_path,
            n_yolos_inference=random_yolos_inference,
            n_llama_inference=random_llama_inference,
            n_bart_inference=random_bart_inference
        )
 
        llama_cont = contention_row["Llama contention "]
        yolos_cont = contention_row["Yolos contention"]
        bart_cont  = contention_row["Bart contention"]
 
        contention = 0

        relative_layer = layer % 400
        if relative_layer <= 1:
            contention = yolos_cont
        elif 2 < relative_layer < 300:
            contention = llama_cont
        elif 300 < relative_layer < 400:
            contention = bart_cont
 
        new_cloud_pending = contention
 
        # Add processing time from cloud nodes
        if len(cloud_nodes) > 0:
            cloud_proc_ms = max(
                self.profiling.get_node_cloud_time(layer, i)
                for i in cloud_nodes
            )
            new_cloud_pending += max(0.0, cloud_proc_ms)
 
        # Special case: all nodes to cloud
        if isAllCloud and len(cloud_nodes) > 0:
            cloud_proc_ms = max(
                self.profiling.get_node_cloud_time(layer, i)
                for i in cloud_nodes
            )
            new_cloud_pending = cloud_proc_ms * self.profiling.numberOfEdgeDevice
 
        return new_cloud_pending
 
    # ================= Next State (Simplified - No Surplus) =================
 
    def get_next_state(self, current_state, action, new_cloud_pending):
        """
        Get next state with simplified format.
       
        State format: [bandwidth, cloud_contention, current_layer, previous_action_pattern]
       
        Args:
            current_state: Current state tuple
            action: Action taken at current layer
            new_cloud_pending: Cloud waiting time for next layer
           
        Returns:
            next_state: New state tuple
            terminal: Whether episode is complete
        """
        _, _, layer, _ = current_state
        layer = int(layer)
       
        # Get current bandwidth
        current_bandwidth = self.get_current_bandwidth()
       
        # Check if this was the last layer
        if layer + 1 < len(self.profiling.layers):
            # terminal = False
            # if 20 < layer < 100:
            #     if np.random.rand() < 0.01:
            #         next_layer = 100
            #     else:
            #         next_layer = layer + 1
 
            # elif 180 < layer < 300:
            #     if np.random.rand() < 0.01:
            #         next_layer = 300
            #     else:
            #         next_layer = layer + 1
 
            # elif 350 < layer < 400:
            #     if np.random.rand() < 0.01:
            #         next_layer = 400
            #         terminal = True
            #     else:
            #       next_layer = layer + 1
 
            # else:
                next_layer = layer + 1
                terminal = False
        else:
            next_layer = layer
            terminal = True
        # print(f"Next layer: {next_layer}, Terminal: {terminal}")
       
        # Convert previous action to pattern for next state
        prev_action_pattern = self._action_to_pattern(action)
       
        # New state: [bandwidth, cloud_contention, next_layer, previous_action_pattern]
        next_state = (
            current_bandwidth,      # Updated bandwidth
            new_cloud_pending,      # Cloud contention for next layer
            next_layer,             # Next layer index
            prev_action_pattern,    # Current action becomes previous for next layer
        )
       
        return next_state, terminal
    
    def _action_to_pattern(self, action):
        """
        Convert action matrix to a pattern representation.
        
        Args:
            action: numpy array of shape (nodes, 2)
            
        Returns:
            tuple: Pattern of node assignments (0=edge, 1=cloud)
        """
        if action is None:
            return None
        return tuple(int(x) for x in action[:, 1])

    # ================= PURE LATENCY COMPUTATION =================

    def compute_latency(self, current_state, current_action, cloud_pending_ms):
        """
        Compute latency (completion time) for the current action.
        
        Returns:
            float: Latency in seconds
        """
        bandwidth, _, layer, prev_action = current_state
        layer = int(layer)

        profiling = self.profiling
        deps = profiling.dependencies

        # ========== 1. TRANSMISSION TIME (Data transfer) ==========
        transmission_times = []

        if prev_action is not None and layer > 0:
            # Data transfer between layers
            prev_assignments = prev_action  # Already a tuple of assignments
            curr_assignments = np.asarray(current_action[:, 1], dtype=int)
            
            # Get number of nodes in previous layer
            prev_layer_nodes = len(prev_assignments)

            for curr_node in range(len(curr_assignments)):
                parent_nodes = deps.get((layer, curr_node), [])
                for (p_layer, p_node) in parent_nodes:
                    # SAFETY CHECK: Ensure parent is from previous layer and index in bounds
                    if p_layer == layer - 1 and p_node < prev_layer_nodes:
                        parent_loc = prev_assignments[p_node]
                    else:
                        # Parent from earlier layer or out of bounds, assume edge
                        parent_loc = 0
                        
                    curr_loc = curr_assignments[curr_node]

                    if parent_loc != curr_loc:
                        # Data must be transmitted
                        output_size = profiling.get_output_size(layer, curr_node)

                        # Transmission time = data size / bandwidth (with RTT floor)
                        transmission_time = max(
                            (output_size) / max(bandwidth, 1e-6),
                            profiling.rtt / 1000.0
                        )
                        transmission_times.append(transmission_time)

        else:
            # First layer: transmit input data to cloud if needed
            for i in range(len(current_action)):
                if current_action[i, 1] == 1:  # Cloud execution
                    transmission_time = max(
                        (profiling.get_input_size()) / max(bandwidth, 1e-6),
                        profiling.rtt / 1000.0
                    )
                    transmission_times.append(transmission_time)

        # Maximum transmission time (parallel transmissions)
        max_transmission_time = max(transmission_times) if transmission_times else 0.0

        # ========== 2. EDGE COMPUTATION TIME ==========
        edge_times = []

        for i in range(len(current_action)):
            if current_action[i, 1] == 0:  # Edge execution
                node_t_s = profiling.get_node_edge_time(layer, i) / 1000.0
                edge_times.append(node_t_s)


       
        edge_total_time_s = sum(edge_times)

        # ========== 3. CLOUD WAITING/IDLE TIME ==========
        actual_idle_time_s = 0.0

        if np.any(current_action[:, 1] == 1):  # Any cloud execution
            cloud_pending_s = cloud_pending_ms / 1000.0
            # Edge may be idle while waiting for cloud
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)

        # ========== 4. TOTAL LATENCY ==========
        # Total latency = edge computation + transmission + idle waiting
        completion_time_s = edge_total_time_s + max_transmission_time + actual_idle_time_s

        # Update cumulative time
        self.cumulative_time_seconds += completion_time_s

        return completion_time_s

    # ================= PURE LATENCY REWARD =================

    def calculate_latency_reward(self, latency_s, scale_factor=100):
        """
        Calculate reward based purely on latency.
        
        Reward = -latency * scale_factor
        Lower latency = Higher reward (less negative)
        
        Args:
            latency_s: Latency in seconds
            scale_factor: Scaling factor for reward magnitude
            
        Returns:
            float: Reward value
        """
        latency_ms = latency_s * scale_factor
        return -latency_ms

    # ================= BACKWARD COMPATIBILITY METHODS =================
    
    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        """
        Backward compatibility method.
        Returns (energy=0, completion_time) tuple.
        """
        completion_time_s = self.compute_latency(
            current_state, current_action, cloud_pending_ms
        )
        return 0.0, completion_time_s  # Energy is always 0

    def calculate_reward(self, layer, total_energy, completion_time_s, **kwargs):
        """
        Simplified reward calculation - ignores deadline/surplus parameters.
        """
        return self.calculate_latency_reward(completion_time_s, scale_factor=1.0)