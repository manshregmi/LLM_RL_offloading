import numpy as np
import pandas as pd
import random
from bisect import bisect_left
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import csv
from datetime import datetime

from profiling.profiling_class import ProfilingData

# ==================== TRACE TRACKER ====================

class TraceTracker:
    def __init__(self, trace_data: List[Tuple[float, float, float]]):
        valid_data = []
        for t, bw, rtt in trace_data:
            if not pd.isna(bw) and not pd.isna(rtt):
                valid_data.append((float(t), float(bw), float(rtt)))
        if not valid_data:
            raise ValueError("No valid trace data provided")
        valid_data.sort(key=lambda x: x[0])

        self.timestamps = np.array([t for t, _, _ in valid_data], dtype=float)
        self.bandwidths = np.array([bw for _, bw, _ in valid_data], dtype=float)
        self.rtts = np.array([rtt for _, _, rtt in valid_data], dtype=float)

        self.min_timestamp = float(self.timestamps[0])
        self.normalized_timestamps = self.timestamps - self.min_timestamp
        self.total_duration = float(self.normalized_timestamps[-1])

        print(f"[OK] Loaded {len(self.timestamps)} trace samples")
        print(f"   Duration: {self.total_duration:.2f}s")
        print(f"   BW Range: {self.bandwidths.min():.1f}-{self.bandwidths.max():.1f} Mbps")
        print(f"   RTT Range: {self.rtts.min():.1f}-{self.rtts.max():.1f} ms")

    def _interpolate_value_at_time(self, time_seconds: float, values: np.ndarray) -> float:
        if self.total_duration > 0:
            query_time = time_seconds % self.total_duration
        else:
            query_time = time_seconds

        if query_time <= self.normalized_timestamps[0]:
            return float(values[0])
        if query_time >= self.normalized_timestamps[-1]:
            return float(values[-1])

        idx = bisect_left(self.normalized_timestamps, query_time)
        t0 = self.normalized_timestamps[idx - 1]
        t1 = self.normalized_timestamps[idx]
        v0 = values[idx - 1]
        v1 = values[idx]
        ratio = (query_time - t0) / (t1 - t0)
        return float(v0 + ratio * (v1 - v0))

    def get_bandwidth_at_time(self, time_seconds: float, use_normalized: bool = False) -> float:
        if use_normalized:
            query_time = float(time_seconds)
        else:
            query_time = float(time_seconds - self.min_timestamp)
        return max(1.0, self._interpolate_value_at_time(query_time, self.bandwidths))

    def get_rtt_at_time(self, time_seconds: float, use_normalized: bool = False) -> float:
        if use_normalized:
            query_time = float(time_seconds)
        else:
            query_time = float(time_seconds - self.min_timestamp)
        return max(0.0, self._interpolate_value_at_time(query_time, self.rtts))


# ==================== CSV LOADERS ====================

def load_trace_data_from_csv(csv_path: str) -> List[Tuple[float, float, float]]:
    df = pd.read_csv(csv_path)
    required = ['timestamp', 'bandwidth_mbps', 'rtt_ms']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['bandwidth_mbps'] = pd.to_numeric(df['bandwidth_mbps'], errors='coerce')
    df['rtt_ms'] = pd.to_numeric(df['rtt_ms'], errors='coerce')
    df = df.dropna(subset=required)
    return list(zip(df['timestamp'].astype(float).values,
                    df['bandwidth_mbps'].astype(float).values,
                    df['rtt_ms'].astype(float).values))


def load_contention_map_from_csv(csv_path: str) -> Dict[Tuple[int, int, int, int], Dict[str, float]]:
    df = pd.read_csv(csv_path)
    key_cols = ['n_llama', 'n_yolos', 'n_bart']
    for col in key_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)

    cont_map = {}
    for _, row in df.iterrows():
        key = (int(row['n_llama']),
               int(row['n_yolos']),
               int(row['n_bart']))
        cont_map[key] = {
            'llama_extra': float(row['llama32_extra']),
            'yolos_extra': float(row['yolos_extra']),
            'bart_extra': float(row['bart_extra']),
        }
    print(f"[OK] Loaded {len(cont_map)} contention combinations")
    return cont_map


# ==================== THREAD STATE ====================

@dataclass
class ThreadState:
    thread_id: int
    assignment: List[int]          # length 13, 0=edge, 1=cloud
    current_layer: int             # -1 means between episodes
    cloud_pending_ms: float        # backlog from previous layers
    episode_count: int
    remaining_time: float          # seconds until current layer finishes
    time_offset: float             # per-thread time offset for trace queries
    total_latency_s: float = 0.0   # cumulative latency over all episodes (seconds)


# ==================== SIMULATOR ====================

class CloudEdgeSimulator:
    def __init__(
        self,
        profiling_data: ProfilingData,
        bandwidth_csv_path: Optional[str] = None,
        contention_csv_path: Optional[str] = None
    ):
        self.profiling = profiling_data

        self.trace_tracker = None
        if bandwidth_csv_path:
            try:
                trace_data = load_trace_data_from_csv(bandwidth_csv_path)
                if trace_data:
                    self.trace_tracker = TraceTracker(trace_data)
            except Exception as e:
                print(f"[WARN] Could not load trace data: {e}")
        self.fallback_bandwidth_mbps = 50.0
        self.fallback_rtt_ms = 10.0

        self.contention_map = None
        if contention_csv_path:
            try:
                self.contention_map = load_contention_map_from_csv(contention_csv_path)
            except Exception as e:
                print(f"[WARN] Could not load contention map: {e}")
        if self.contention_map is None:
            self.contention_map = {
                (0,0,0): {
                    'llama_extra': 0.0,
                    'yolos_extra': 0.0,
                    'bart_extra': 0.0
                }
            }

        self.i = 0
        self.j = 0
        self.k = 0
        self.cumulative_time_seconds = 0.0

        if self.trace_tracker:
            max_time = self.trace_tracker.total_duration
            self.episode_offset = random.uniform(0, max_time * 0.9) if max_time > 0 else 0.0
        else:
            self.episode_offset = 0.0

        print(f"[CONFIG] Continuous system initialised. Trace offset: {self.episode_offset:.2f}s")

        self.tft_bw = None
        self.tft_rtt = None
        self.max_encoder_length = 32

    # ========== Helper: get bandwidth/rtt at a specific global time ==========
    def _get_bandwidth_at_global_time(self, global_time: float) -> float:
        if self.trace_tracker:
            query_time = float(global_time + self.episode_offset)
            return self.trace_tracker.get_bandwidth_at_time(query_time, use_normalized=True)
        return self.fallback_bandwidth_mbps

    def _get_rtt_at_global_time(self, global_time: float) -> float:
        if self.trace_tracker:
            query_time = float(global_time + self.episode_offset)
            return self.trace_tracker.get_rtt_at_time(query_time, use_normalized=True)
        return self.fallback_rtt_ms

    def get_current_bandwidth_mbps(self) -> float:
        return self._get_bandwidth_at_global_time(self.cumulative_time_seconds)
    

    def get_current_rtt_ms(self) -> float:
        return self._get_rtt_at_global_time(self.cumulative_time_seconds)

    def get_possible_actions(self, layer):
        if layer >= len(self.profiling.layers):
            return []
        nodes = self.profiling.get_num_nodes(layer)
        actions = []
        for pattern in range(2 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1
            actions.append(a)
        return actions

    def get_next_state_cloud_waiting_time(self, layer, current_action, isAllCloud=False):
        layer = int(layer)
        key = (self.i, self.j, self.k)
        row = self.contention_map.get(key)
        if row is None:
            llama_extra = yolos_extra = bart_extra = 0.0
        else:
            llama_extra = row['llama_extra']
            yolos_extra = row['yolos_extra']
            bart_extra = row['bart_extra']


        contention_ms = max(0.0, min(3.0, contention_ms))

        cloud_nodes = np.where(current_action[:, 1] == 1)[0]
        if len(cloud_nodes) == 0:
            return 0.0

        effective_cloud_times = []

        for node_idx in cloud_nodes:

            model_idx = self._get_model_idx(layer, node_idx)

            if model_idx == 0:      # LLaMA
                contention_extra = llama_extra

            elif model_idx == 1:    # YOLOS
                contention_extra = yolos_extra

            elif model_idx == 2:    # BART
                contention_extra = bart_extra

            else:
                contention_extra = 0.0

            cloud_proc_ms = self.profiling.get_node_cloud_time(
                layer,
                node_idx
            )

            effective_cloud_times.append(
                max(0.0, cloud_proc_ms)
                + max(0.0, contention_extra)
            )
            new_cloud_pending = contention_ms

        if len(cloud_nodes) > 0:
            cloud_proc_ms = max(self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
            new_cloud_pending += max(0.0, cloud_proc_ms)

        if isAllCloud and len(cloud_nodes) > 0:
            cloud_proc_ms = max(self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
            new_cloud_pending = (contention_ms + cloud_proc_ms) * self.profiling.numberOfEdgeDevice

        return new_cloud_pending

    def get_next_state(self, current_state, action, new_cloud_pending):
        _, _, layer, _ = current_state
        layer = int(layer)
        bandwidth_mbps = self.get_current_bandwidth_mbps()
        terminal = False
        if layer + 1 < len(self.profiling.layers):
            next_layer = layer + 1
        else:
            terminal = True
            next_layer = layer
        prev_action_pattern = tuple(int(x) for x in action[:, 1])
        next_state = (bandwidth_mbps, new_cloud_pending, next_layer, prev_action_pattern)
        return next_state, terminal

    def compute_latency(self, current_state, current_action, cloud_pending_ms):
        # Kept for backward compatibility; not used in concurrent simulation
        bandwidth_mbps, _, layer, prev_action_pattern = current_state
        layer = int(layer)
        rtt_ms = self.get_current_rtt_ms()
        rtt_s = rtt_ms / 1000.0

        profiling = self.profiling
        deps = profiling.dependencies

        transmission_times = []

        if prev_action_pattern is not None and layer > 0:
            prev_assignments = prev_action_pattern
            curr_assignments = np.asarray(current_action[:, 1], dtype=int)
            for curr_node in range(len(curr_assignments)):
                parent_nodes = deps.get((layer, curr_node), [])
                for (p_layer, p_node) in parent_nodes:
                    if p_layer == layer - 1:
                        parent_loc = prev_assignments[p_node]
                    else:
                        parent_loc = 0
                    curr_loc = curr_assignments[curr_node]
                    if parent_loc != curr_loc:
                        output_size_kb = profiling.get_output_size(layer, curr_node)
                        tx_time = (output_size_kb * 8) / (bandwidth_mbps * 1000.0)
                        transmission_time = max(tx_time, rtt_s)
                        transmission_times.append(transmission_time)
        else:
            for i in range(len(current_action)):
                if current_action[i, 1] == 1:
                    input_size_kb = profiling.get_input_size()
                    tx_time = (input_size_kb * 8) / (bandwidth_mbps * 1000.0)
                    transmission_time = max(tx_time, rtt_s)
                    transmission_times.append(transmission_time)

        max_transmission_time = max(transmission_times) if transmission_times else 0.0

        edge_times = []
        for i in range(len(current_action)):
            if current_action[i, 1] == 0:
                node_t_s = profiling.get_node_edge_time(layer, i) / 1000.0
                edge_times.append(node_t_s)

        if layer in [3, 5]:
            edge_total_time_s = max(edge_times) if edge_times else 0.0
        else:
            edge_total_time_s = sum(edge_times)

        actual_idle_time_s = 0.0
        if np.any(current_action[:, 1] == 1):
            cloud_pending_s = cloud_pending_ms / 1000.0
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)

        completion_time_s = edge_total_time_s + max_transmission_time + actual_idle_time_s
        self.cumulative_time_seconds += completion_time_s
        return completion_time_s

    # ========== Isolated layer duration ==========
    def compute_layer_duration(
        self,
        layer_idx: int,
        action_matrix: np.ndarray,
        cloud_pending_ms: float,
        prev_action_pattern: Optional[Tuple[int, ...]],
        bandwidth: float,
        rtt_ms: float,
        # contention_extra: float
        contention_map: Dict[Tuple[int, int], float]
    ) -> Tuple[float, float]:
        profiling = self.profiling
        deps = profiling.dependencies
        layer = layer_idx
        num_nodes = len(profiling.layers[layer])

        transmission_times = []
        if prev_action_pattern is not None and layer > 0:
            prev_assignments = prev_action_pattern
            curr_assignments = action_matrix[:, 1]
            for curr_node in range(num_nodes):
                parent_nodes = deps.get((layer, curr_node), [])
                for (p_layer, p_node) in parent_nodes:
                    if p_layer == layer - 1:
                        parent_loc = prev_assignments[p_node]
                    else:
                        parent_loc = 0
                    curr_loc = curr_assignments[curr_node]
                    if parent_loc != curr_loc:
                        output_size_mb = profiling.get_output_size(layer, curr_node)
                        tx_time = (output_size_mb ) / (bandwidth)
                        rtt_s = rtt_ms / 1000.0
                        transmission_time = max(tx_time, rtt_s)
                        transmission_times.append(transmission_time)
        else:
            for i in range(num_nodes):
                if action_matrix[i, 1] == 1:
                    input_size_mb = profiling.get_input_size()
                    tx_time = (input_size_mb) / (bandwidth)
                    rtt_s = rtt_ms / 1000.0
                    transmission_time = max(tx_time, rtt_s)
                    transmission_times.append(transmission_time)

        max_transmission_time = max(transmission_times) if transmission_times else 0.0

        edge_times = []
        for i in range(num_nodes):
            if action_matrix[i, 1] == 0:
                node_t_s = profiling.get_node_edge_time(layer, i) / 1000.0
                edge_times.append(node_t_s)

        # if layer in [3, 5]:
        #     edge_total_time_s = max(edge_times) if edge_times else 0.0
        # else:
        #     edge_total_time_s = sum(edge_times)
        edge_total_time_s = sum(edge_times)
        actual_idle_time_s = 0.0
        if np.any(action_matrix[:, 1] == 1):
            cloud_pending_s = cloud_pending_ms / 1000.0
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)

        completion_time_s = edge_total_time_s + max_transmission_time + actual_idle_time_s

        # Update cloud_pending_ms for next layer
        if np.any(action_matrix[:, 1] == 1):
            cloud_nodes = np.where(action_matrix[:, 1] == 1)[0]
            effective_cloud_times_ms =[]
            for node_idx in cloud_nodes:
                cloud_proc_ms = max(profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
                contention_extra_ms = contention_map.get((layer,node_idx),0.0)
                effective_cloud_times_ms.append(max(0.0, cloud_proc_ms)+ max(0.0, contention_extra_ms))
            cloud_time_ms = max(effective_cloud_times_ms)                                    
            edge_times_ms = [profiling.get_node_edge_time(layer, i) for i in range(num_nodes) if action_matrix[i, 1] == 0]
            edge_time_ms = sum(edge_times_ms)
            new_cloud_pending = max(0.0, cloud_pending_ms - edge_time_ms)
            new_cloud_pending += cloud_proc_ms 
        else:
            new_cloud_pending = 0.0

        return completion_time_s, new_cloud_pending

    # ========== Contention refresh ==========
    def _get_model_idx(self, layer, node):
        """
        Return model index (0=Llama, 1=Yolos, 2=Bart) for a given (layer, node).
        Uses model_boundary_layers extended with model tuples if available.
        Fallback to old layer‑based logic for backward compatibility.
        """
        layer = int(layer)
        node = int(node)
        # Try the extended 4‑tuple format (start, end, segments, models)
        if self.profiling.model_boundary_layers and len(self.profiling.model_boundary_layers[0]) == 4:
            for start, end, segments, models in self.profiling.model_boundary_layers:
                if start <= layer <= end:
                    if node < len(models):
                        return models[node]
                    return models[0] if models else 0

    def _refresh_contention_from_threads(self, threads: List[ThreadState]):
        raw_i = raw_j = raw_k = 0

        # layer_mapping = {
        #     0: [(0, 'i')],
        #     1: [(1, 'i')],
        #     2: [(2, 'i')],
        #     3: [(3, 'j'), (4, 'k'), (5, 'l')],
        #     4: [(6, 'j'), (7, 'k'), (8, 'l')],
        #     5: [(9, 'j'), (10, 'k'), (11, 'l')],
        #     6: []   # v13 forced edge
        # }

        # for thread in threads:
        #     if thread.current_layer == -1:
        #         continue
        #     layer = thread.current_layer
        #     if layer not in layer_mapping:
        #         continue
        #     for node_idx, ctype in layer_mapping[layer]:
        #         if thread.assignment[node_idx] == 1:
        #             if ctype == 'i':
        #                 raw_i += 1
        #             elif ctype == 'j':
        #                 raw_j += 1
        #             elif ctype == 'k':
        #                 raw_k += 1
        for thread in threads:
            if thread.current_layer == -1:
                continue
            layer = thread.current_layer
            num_nodes = len(self.profiling.layers[layer])
            # Flattened start index for this layer
            flat_start = sum(len(self.profiling.layers[l]) for l in range(layer))

            for node_idx in range(num_nodes):
                # Check if this node is offloaded to cloud (assignment value = 1)
                if thread.assignment[flat_start + node_idx] == 1:
                    model_idx = self._get_model_idx(layer, node_idx)
                    if model_idx == 0:      # Llama
                        raw_i += 1
                    elif model_idx == 1:    # Yolos
                        raw_j += 1
                    elif model_idx == 2:    # Bart
                        raw_k += 1
                    else:
                        # Fallback: treat unknown as Llama
                        raw_i += 1

        self.i = min(raw_i, 3)
        self.j = min(raw_j, 3)
        self.k = min(raw_k, 3)

    # ========== TFT Integration ==========

    # def load_tft_models(self, bw_model_dir: Path, rtt_model_dir: Path):
    #     from simulator.tft_inference import TFTInference
    #     self.tft_bw = TFTInference(bw_model_dir)
    #     self.tft_rtt = TFTInference(rtt_model_dir)
    #     self.max_encoder_length = self.tft_bw.max_encoder_length

    # def _get_tft_context_window(self) -> pd.DataFrame:
    #     max_len = self.max_encoder_length
    #     current_step = int(self.cumulative_time_seconds)
    #     start_step = max(0, current_step - max_len + 1)

    #     steps = []
    #     bw_vals = []
    #     rtt_vals = []
    #     for step in range(start_step, current_step + 1):
    #         query_time = float(step) + self.episode_offset
    #         if step == current_step:
    #             bw = self.get_current_bandwidth_mbps()
    #             rtt = self.get_current_rtt_ms()
    #         else:
    #             bw = self.trace_tracker.get_bandwidth_at_time(query_time, use_normalized=True)
    #             rtt = self.trace_tracker.get_rtt_at_time(query_time, use_normalized=True)
    #         steps.append(step)
    #         bw_vals.append(bw)
    #         rtt_vals.append(rtt)

    #     while len(steps) < max_len:
    #         steps.insert(0, steps[0] - 1)
    #         bw_vals.insert(0, bw_vals[0])
    #         rtt_vals.insert(0, rtt_vals[0])

    #     df = pd.DataFrame({
    #         "step": steps[-max_len:],
    #         "bandwidth_mbps": bw_vals[-max_len:],
    #         "rtt_ms": rtt_vals[-max_len:]
    #     })
    #     return df

    # def get_predicted_bw_rtt(self) -> Tuple[float, float]:
    #     if self.tft_bw is None or self.tft_rtt is None:
    #         raise RuntimeError("TFT models not loaded. Call load_tft_models first.")
    #     try:
    #         df_window = self._get_tft_context_window()
    #         bw_pred = self.tft_bw.predict_from_window(df_window)
    #         rtt_pred = self.tft_rtt.predict_from_window(df_window)
    #         return bw_pred, rtt_pred
    #     except Exception as e:
    #         print(f"[WARN] TFT prediction failed: {e}. Falling back to trace values.")
    #         return self.get_current_bandwidth_mbps(), self.get_current_rtt_ms()

    def get_contention_extra_map(self) -> Dict[Tuple[int, int], float]:
        key = (self.i, self.j, self.k)
        row = self.contention_map.get(key)
        key = (self.i, self.j, self.k)
        row = self.contention_map.get(key)
        if row is None:
            llama_extra = yolos_extra = bart_extra = 0.0
        else:
            llama_extra = row.get('llama_extra', 0.0)
            yolos_extra = row.get('yolos_extra', 0.0)
            bart_extra = row.get('bart_extra', 0.0)

        contention_map = {}
        for l_idx, layer in enumerate(self.profiling.layers):
            for n_idx in range(len(layer)):
                model_idx = self._get_model_idx(l_idx, n_idx)
                if model_idx == 0:
                    extra = llama_extra
                elif model_idx == 1:
                    extra = yolos_extra
                elif model_idx == 2:
                    extra = bart_extra
                else:
                    extra = 0.0
                contention_map[(l_idx, n_idx)] = extra
        return contention_map

    # ========== Concurrent Simulation with per-thread latency tracking ==========
    def simulate_concurrent(
        self,
        num_threads: int = 10,
        episodes_per_thread: int = 10000,
        use_tft: bool = True
    ):
        print(
            f"[START] Starting concurrent simulation with "
            f"{num_threads} threads, {episodes_per_thread} episodes each."
        )

        # ============================================================
        # Open CSV file for contention trace logging
        # ============================================================

        csv_filename = (
            f"contention_log_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            'timestamp_s',
            'bandwidth_mbps',
            'rtt_ms',
            'i',
            'j',
            'k',
            'llama_extra_ms',
            'yolos_extra_ms',
            'bart_extra_ms',
            'episodes_completed',
            'total_episodes'
        ])

        print(f"[FILE] Logging contention data to {csv_filename}")

        # ============================================================
        # Determine maximum random start offset
        # ============================================================

        if self.trace_tracker:
            max_offset = self.trace_tracker.total_duration * 0.9
        else:
            max_offset = 10.0

        # ============================================================
        # Initialise concurrent threads
        # ============================================================

        threads: List[ThreadState] = []

        for tid in range(num_threads):

            offset = random.uniform(0, max_offset)

            self.i, self.j, self.k = 0, 0, 0

            contention_map = self.get_contention_extra_map()

            local_time = offset

            if use_tft and self.tft_bw is not None:

                # TFT bandwidth prediction is assumed to be Mbps,
                # so convert to MB/s before passing it to the DP solver.
                bw_mbps, rtt = self.get_predicted_bw_rtt()
                bw = bw_mbps / 8.0

            else:

                bw = self._get_bandwidth_at_global_time(
                    local_time
                )

                rtt = self._get_rtt_at_global_time(
                    local_time
                )

            from simulator.shortest_path_solver import (
                run_dijkstra_partition
            )

            assignment, _ = run_dijkstra_partition(
                self.profiling,
                bw,
                rtt,
                contention_map
            )
            # if tid == 0:
            #     print("[DEBUG] DP assignment for thread 0:")
            #     print(assignment)
            if tid == 0:
                cloud_indices = [
                    idx
                    for idx, value in enumerate(assignment)
                    if value == 1
                ]

                print(
                    f"[DEBUG] Cloud nodes for thread 0: "
                    f"{cloud_indices}"
                )

                print(
                    f"[DEBUG] Total cloud nodes: "
                    f"{len(cloud_indices)}/{len(assignment)}"
                )


            thread = ThreadState(
                thread_id=tid,
                assignment=assignment,
                current_layer=-1,
                cloud_pending_ms=0.0,
                episode_count=0,
                remaining_time=0.0,
                time_offset=offset,
                total_latency_s=0.0
            )

            threads.append(thread)

        # ============================================================
        # Initialise simulation counters
        # ============================================================

        completed_threads = 0
        total_episodes_completed = 0

        # Used only for terminal progress printing.
        last_logged_episode = -1

        # Used for contention-trace CSV.
        # None guarantees that the initial contention state is recorded.
        last_logged_contention = None

        # ============================================================
        # Start all threads at layer 0
        # ============================================================

        for thread in threads:
            thread.current_layer = 0

        self._refresh_contention_from_threads(threads)

        # ============================================================
        # Write the initial contention state at t = 0
        # ============================================================

        current_contention = (
            self.i,
            self.j,
            self.k
        )

        bw_real = self.get_current_bandwidth_mbps()
        rtt_real = self.get_current_rtt_ms()

        row = self.contention_map.get(
            current_contention
        )

        if row is None:
            llama_extra = 0.0
            yolo_extra = 0.0
            bart_extra = 0.0
        else:
            llama_extra = row.get(
                'llama_extra',
                0.0
            )
            yolo_extra = row.get(
                'yolos_extra',
                0.0
            )
            bart_extra = row.get(
                'bart_extra',
                0.0
            )

        csv_writer.writerow([
            self.cumulative_time_seconds,
            bw_real,
            rtt_real,
            self.i,
            self.j,
            self.k,
            llama_extra,
            yolo_extra,
            bart_extra,
            total_episodes_completed,
            num_threads * episodes_per_thread
        ])

        csv_file.flush()

        last_logged_contention = current_contention

        # ============================================================
        # Compute initial layer durations
        # ============================================================

        for thread in threads:

            local_time = (
                self.cumulative_time_seconds
                + thread.time_offset
            )

            
            bw = self._get_bandwidth_at_global_time(
                local_time
            )

            rtt = self._get_rtt_at_global_time(
                local_time
            )

            contention_map = (
                self.get_contention_extra_map()
            )

            layer = 0

            num_nodes = len(
                self.profiling.layers[layer]
            )

            action = np.zeros(
                (num_nodes, 2),
                dtype=int
            )

            action[:, 0] = layer

            flat_idx = 0

            for n_idx in range(num_nodes):
                action[n_idx, 1] = (
                    thread.assignment[
                        flat_idx + n_idx
                    ]
                )

            extra = 0.0

            if np.any(action[:, 1] == 1):

                cloud_nodes = np.where(
                    action[:, 1] == 1
                )[0]

                # Use the maximum contention penalty among
                # cloud nodes in this layer.
                # extra = max(
                #     contention_map.get(
                #         (layer, node_idx),
                #         0.0
                #     )
                #     for node_idx in cloud_nodes
                # )

            duration, new_pending = (
                self.compute_layer_duration(
                    layer,
                    action,
                    thread.cloud_pending_ms,
                    None,
                    bw,
                    rtt,
                    contention_map
                )
            )

            thread.remaining_time = duration
            thread.cloud_pending_ms = new_pending

            # Accumulate request latency for this thread.
            thread.total_latency_s += duration

        # ============================================================
        # Main event-driven concurrent simulation
        # ============================================================

        while completed_threads < num_threads:

            # --------------------------------------------------------
            # Find next layer-completion event
            # --------------------------------------------------------

            min_time = float('inf')

            for t in threads:

                if (
                    t.episode_count < episodes_per_thread
                    and t.remaining_time > 0
                ):
                    if t.remaining_time < min_time:
                        min_time = t.remaining_time

            if min_time == float('inf'):
                break

            # Advance global simulated time.
            self.cumulative_time_seconds += min_time

            # Advance every running thread by the same amount.
            for t in threads:

                if (
                    t.episode_count < episodes_per_thread
                    and t.remaining_time > 0
                ):

                    t.remaining_time -= min_time

                    if t.remaining_time < 1e-9:
                        t.remaining_time = 0.0

            # --------------------------------------------------------
            # Process every thread that completed its current layer
            # --------------------------------------------------------

            for t in threads:

                if t.episode_count >= episodes_per_thread:
                    continue

                if t.remaining_time != 0.0:
                    continue

                # ====================================================
                # Current episode completed
                # ====================================================

                if (
                    t.current_layer
                    == len(self.profiling.layers) - 1
                ):

                    t.episode_count += 1
                    total_episodes_completed += 1

                    # ------------------------------------------------
                    # Thread has completed all requested episodes
                    # ------------------------------------------------

                    if t.episode_count >= episodes_per_thread:

                        completed_threads += 1

                        t.current_layer = -1
                        t.remaining_time = 0.0

                        print(
                            f"[OK] Thread {t.thread_id} "
                            f"completed "
                            f"{episodes_per_thread} episodes."
                        )

                        continue

                    # ------------------------------------------------
                    # Start next episode
                    # ------------------------------------------------

                    t.current_layer = -1

                    self._refresh_contention_from_threads(
                        threads
                    )

                    contention_map = (
                        self.get_contention_extra_map()
                    )

                    local_time = (
                        self.cumulative_time_seconds
                        + t.time_offset
                    )

                    if use_tft and self.tft_bw is not None:

                        # TFT gives Mbps -> convert to MB/s.
                        bw_mbps, rtt = (
                            self.get_predicted_bw_rtt()
                        )

                        bw = bw_mbps / 8.0

                    else:

                        bw = (
                            self._get_bandwidth_at_global_time(
                                local_time
                            )
                        )

                        rtt = (
                            self._get_rtt_at_global_time(
                                local_time
                            )
                        )

                    from simulator.shortest_path_solver import (
                        run_dijkstra_partition
                    )

                    assignment, _ = (
                        run_dijkstra_partition(
                            self.profiling,
                            bw,
                            rtt,
                            contention_map
                        )
                    )

                    t.assignment = assignment
                    t.cloud_pending_ms = 0.0
                    t.current_layer = 0

                    # Recalculate contention now that this
                    # thread has entered layer 0.
                    self._refresh_contention_from_threads(
                        threads
                    )

                    local_time = (
                        self.cumulative_time_seconds
                        + t.time_offset
                    )

                    bw = (
                        self._get_bandwidth_at_global_time(
                            local_time
                        )
                    )

                    rtt = (
                        self._get_rtt_at_global_time(
                            local_time
                        )
                    )

                    contention_map = (
                        self.get_contention_extra_map()
                    )

                    layer = 0

                    num_nodes = len(
                        self.profiling.layers[layer]
                    )

                    action = np.zeros(
                        (num_nodes, 2),
                        dtype=int
                    )

                    action[:, 0] = layer

                    flat_idx = 0

                    for n_idx in range(num_nodes):

                        action[n_idx, 1] = (
                            t.assignment[
                                flat_idx + n_idx
                            ]
                        )

                    extra = 0.0

                    if np.any(action[:, 1] == 1):

                        cloud_nodes = np.where(
                            action[:, 1] == 1
                        )[0]

                        # extra = max(
                        #     contention_map.get(
                        #         (layer, node_idx),
                        #         0.0
                        #     )
                        #     for node_idx in cloud_nodes
                        # )

                    duration, new_pending = (
                        self.compute_layer_duration(
                            layer,
                            action,
                            t.cloud_pending_ms,
                            None,
                            bw,
                            rtt,
                            contention_map
                        )
                    )

                    t.remaining_time = duration
                    t.cloud_pending_ms = new_pending

                    t.total_latency_s += duration

                # ====================================================
                # Move to next layer in current episode
                # ====================================================

                else:

                    next_layer = t.current_layer + 1

                    t.current_layer = next_layer

                    self._refresh_contention_from_threads(
                        threads
                    )

                    local_time = (
                        self.cumulative_time_seconds
                        + t.time_offset
                    )

                    bw = (
                        self._get_bandwidth_at_global_time(
                            local_time
                        )
                    )

                    rtt = (
                        self._get_rtt_at_global_time(
                            local_time
                        )
                    )

                    contention_map = (
                        self.get_contention_extra_map()
                    )

                    layer = next_layer

                    num_nodes = len(
                        self.profiling.layers[layer]
                    )

                    action = np.zeros(
                        (num_nodes, 2),
                        dtype=int
                    )

                    action[:, 0] = layer

                    flat_idx = sum(
                        len(self.profiling.layers[l])
                        for l in range(layer)
                    )

                    for n_idx in range(num_nodes):

                        action[n_idx, 1] = (
                            t.assignment[
                                flat_idx + n_idx
                            ]
                        )

                    prev_layer = layer - 1

                    prev_num = len(
                        self.profiling.layers[
                            prev_layer
                        ]
                    )

                    prev_flat = sum(
                        len(self.profiling.layers[l])
                        for l in range(prev_layer)
                    )

                    prev_action_pattern = tuple(
                        t.assignment[
                            prev_flat + i
                        ]
                        for i in range(prev_num)
                    )

                    # extra = 0.0

                    if np.any(action[:, 1] == 1):

                        cloud_nodes = np.where(
                            action[:, 1] == 1
                        )[0]

                        # extra = max(
                        #     contention_map.get(
                        #         (layer, node_idx),
                        #         0.0
                        #     )
                        #     for node_idx in cloud_nodes
                        # )

                    duration, new_pending = (
                        self.compute_layer_duration(
                            layer,
                            action,
                            t.cloud_pending_ms,
                            prev_action_pattern,
                            bw,
                            rtt,
                            contention_map
                        )
                    )

                    t.remaining_time = duration
                    t.cloud_pending_ms = new_pending

                    t.total_latency_s += duration

            # ========================================================
            # Refresh final contention state after processing all
            # simultaneous completions at this simulation timestamp.
            #
            # This is important because several threads may finish
            # layers at exactly the same event time.
            # ========================================================

            self._refresh_contention_from_threads(
                threads
            )

            current_contention = (
                self.i,
                self.j,
                self.k
            )

            # ========================================================
            # CONTENTION TRACE LOGGING
            #
            # Write whenever (i,j,k) changes.
            # This is independent of episode count.
            # ========================================================

            if (
                current_contention
                != last_logged_contention
            ):

                bw_real = (
                    self.get_current_bandwidth_mbps()
                )

                rtt_real = (
                    self.get_current_rtt_ms()
                )

                row = self.contention_map.get(
                    current_contention
                )

                if row is None:

                    llama_extra = 0.0
                    yolo_extra = 0.0
                    bart_extra = 0.0

                else:

                    llama_extra = row.get(
                        'llama_extra',
                        0.0
                    )

                    yolo_extra = row.get(
                        'yolos_extra',
                        0.0
                    )

                    bart_extra = row.get(
                        'bart_extra',
                        0.0
                    )

                csv_writer.writerow([
                    self.cumulative_time_seconds,
                    bw_real,
                    rtt_real,
                    self.i,
                    self.j,
                    self.k,
                    llama_extra,
                    yolo_extra,
                    bart_extra,
                    total_episodes_completed,
                    num_threads * episodes_per_thread
                ])

                csv_file.flush()

                last_logged_contention = (
                    current_contention
                )

            # ========================================================
            # TERMINAL PROGRESS PRINTING
            #
            # Print once every 10 completed episodes.
            # This no longer controls CSV trace logging.
            # ========================================================

            if (
                total_episodes_completed > 0
                and total_episodes_completed % 10 == 0
                and total_episodes_completed
                != last_logged_episode
            ):

                bw_real = (
                    self.get_current_bandwidth_mbps()
                )

                rtt_real = (
                    self.get_current_rtt_ms()
                )

                key = (
                    self.i,
                    self.j,
                    self.k
                )

                row = self.contention_map.get(key)

                if row is None:

                    llama_extra = 0.0
                    yolo_extra = 0.0
                    bart_extra = 0.0

                else:

                    llama_extra = row.get(
                        'llama_extra',
                        0.0
                    )

                    yolo_extra = row.get(
                        'yolos_extra',
                        0.0
                    )

                    bart_extra = row.get(
                        'bart_extra',
                        0.0
                    )

                print(
                    f"[PROGRESS] "
                    f"T={self.cumulative_time_seconds:.2f}s | "
                    f"BW={bw_real:.1f}Mbps "
                    f"RTT={rtt_real:.1f}ms | "
                    f"Contention "
                    f"({self.i},{self.j},{self.k}) | "
                    f"Extras: "
                    f"L={llama_extra:.2f}ms, "
                    f"Y={yolo_extra:.2f}ms, "
                    f"B={bart_extra:.2f}ms | "
                    f"Episodes: "
                    f"{total_episodes_completed}/"
                    f"{num_threads * episodes_per_thread}"
                )

                last_logged_episode = (
                    total_episodes_completed
                )

        # ============================================================
        # Close contention trace
        # ============================================================

        csv_file.close()

        print(
            f"[FILE] Contention data saved to "
            f"{csv_filename}"
        )

        # ============================================================
        # Final simulation summary
        # ============================================================

        total_seconds = (
            self.cumulative_time_seconds
        )

        total_ms = total_seconds * 1000.0

        # Actual average request/pipeline latency.
        # Do NOT divide concurrent makespan by number of episodes.
        total_episode_latency_s = sum(
            t.total_latency_s
            for t in threads
        )

        total_episode_count = (
            num_threads * episodes_per_thread
        )

        overall_avg_ms = (
            total_episode_latency_s
            / total_episode_count
        ) * 1000.0

        print(
            f"\n[FINISH] All threads completed "
            f"{episodes_per_thread} episodes."
        )

        print(
            f"Total simulation time: "
            f"{total_seconds:.2f}s "
            f"({total_ms:.2f} ms)"
        )

        print(
            f"Overall average latency per episode: "
            f"{overall_avg_ms:.3f} ms"
        )

        # ============================================================
        # Per-thread statistics
        # ============================================================

        print(
            "\n[STATS] Per-thread average "
            "latencies (ms):"
        )

        for t in threads:

            avg_ms = (
                t.total_latency_s
                / episodes_per_thread
            ) * 1000.0

            print(
                f"  Thread {t.thread_id}: "
                f"{avg_ms:.3f} ms"
            )
    # def simulate_concurrent(
    #     self,
    #     num_threads: int = 10,
    #     episodes_per_thread: int = 10000,
    #     use_tft: bool = True
    # ):
    #     print(f"[START] Starting concurrent simulation with {num_threads} threads, {episodes_per_thread} episodes each.")

    #     # ---- Open CSV file for logging ----
    #     csv_filename = f"contention_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    #     csv_file = open(csv_filename, 'w', newline='')
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow([
    #         'timestamp_s', 'bandwidth_mbps', 'rtt_ms',
    #         'i', 'j', 'k',
    #         'llama_extra_ms', 'yolos_extra_ms', 'bart_extra_ms',
    #         'episodes_completed', 'total_episodes'
    #     ])
    #     print(f"[FILE] Logging contention data to {csv_filename}")

    #     # Determine max offset for thread start times
    #     if self.trace_tracker:
    #         max_offset = self.trace_tracker.total_duration * 0.9
    #     else:
    #         max_offset = 10.0

    #     # Initialise threads with random time offsets
    #     threads: List[ThreadState] = []
    #     for tid in range(num_threads):
    #         offset = random.uniform(0, max_offset)
    #         self.i, self.j, self.k = 0, 0, 0
    #         contention_map = self.get_contention_extra_map()
    #         local_time = 0.0 + offset
    #         if use_tft and self.tft_bw is not None:
    #             bw, rtt = self.get_predicted_bw_rtt()
    #         else:
    #             bw = self._get_bandwidth_MBps_at_global_time(local_time)
    #             rtt = self._get_rtt_at_global_time(local_time)

    #         from simulator.shortest_path_solver import run_dijkstra_partition
    #         assignment, _ = run_dijkstra_partition(
    #             self.profiling, bw, rtt, contention_map
    #         )

    #         thread = ThreadState(
    #             thread_id=tid,
    #             assignment=assignment,
    #             current_layer=-1,
    #             cloud_pending_ms=0.0,
    #             episode_count=0,
    #             remaining_time=0.0,
    #             time_offset=offset,
    #             total_latency_s=0.0
    #         )
    #         threads.append(thread)

    #     # Start all threads at layer 0
    #     for thread in threads:
    #         thread.current_layer = 0
    #     self._refresh_contention_from_threads(threads)

    #     # Compute initial layer durations using per-thread local time
    #     for thread in threads:
    #         local_time = self.cumulative_time_seconds + thread.time_offset
    #         bw = self._get_bandwidth_MBps_at_global_time(local_time)
    #         rtt = self._get_rtt_at_global_time(local_time)
    #         contention_map = self.get_contention_extra_map()
    #         layer = 0
    #         num_nodes = len(self.profiling.layers[layer])
    #         action = np.zeros((num_nodes, 2), dtype=int)
    #         action[:, 0] = layer
    #         flat_idx = 0
    #         for n_idx in range(num_nodes):
    #             action[n_idx, 1] = thread.assignment[flat_idx + n_idx]
    #         extra = 0.0
    #         if np.any(action[:, 1] == 1):
    #             cloud_nodes = np.where(action[:, 1] == 1)[0]
    #             key = (layer, cloud_nodes[0])
    #             extra = contention_map.get(key, 0.0)
    #         duration, new_pending = self.compute_layer_duration(
    #             layer, action, thread.cloud_pending_ms, None, bw, rtt, extra
    #         )
    #         thread.remaining_time = duration
    #         thread.cloud_pending_ms = new_pending
    #         # Accumulate latency
    #         thread.total_latency_s += duration

    #     completed_threads = 0
    #     total_episodes_completed = 0

    #     while completed_threads < num_threads:
    #         # Find minimum remaining time
    #         min_time = float('inf')
    #         for t in threads:
    #             if t.episode_count < episodes_per_thread and t.remaining_time > 0:
    #                 if t.remaining_time < min_time:
    #                     min_time = t.remaining_time
    #         if min_time == float('inf'):
    #             break

    #         self.cumulative_time_seconds += min_time

    #         for t in threads:
    #             if t.episode_count < episodes_per_thread and t.remaining_time > 0:
    #                 t.remaining_time -= min_time
    #                 if t.remaining_time < 1e-9:
    #                     t.remaining_time = 0.0

    #         # Process threads that finished a layer
    #         for t in threads:
    #             if t.episode_count >= episodes_per_thread:
    #                 continue
    #             if t.remaining_time == 0.0:
    #                 if t.current_layer == len(self.profiling.layers) - 1:
    #                     # Episode complete
    #                     t.episode_count += 1
    #                     total_episodes_completed += 1
    #                     if t.episode_count >= episodes_per_thread:
    #                         completed_threads += 1
    #                         t.current_layer = -1
    #                         t.remaining_time = 0.0
    #                         print(f"[OK] Thread {t.thread_id} completed {episodes_per_thread} episodes.")
    #                         continue
    #                     else:
    #                         # Start new episode
    #                         t.current_layer = -1
    #                         self._refresh_contention_from_threads(threads)
    #                         contention_map = self.get_contention_extra_map()
    #                         local_time = self.cumulative_time_seconds + t.time_offset
    #                         if use_tft and self.tft_bw is not None:
    #                             bw, rtt = self.get_predicted_bw_rtt()
    #                         else:
    #                             bw = self._get_bandwidth_MBps_at_global_time(local_time)
    #                             rtt = self._get_rtt_at_global_time(local_time)
    #                         from simulator.shortest_path_solver import run_dijkstra_partition
    #                         assignment, _ = run_dijkstra_partition(
    #                             self.profiling, bw, rtt, contention_map
    #                         )
    #                         t.assignment = assignment
    #                         t.cloud_pending_ms = 0.0
    #                         t.current_layer = 0
    #                         self._refresh_contention_from_threads(threads)
    #                         local_time = self.cumulative_time_seconds + t.time_offset
    #                         bw = self._get_bandwidth_at_global_time(local_time)
    #                         rtt = self._get_rtt_at_global_time(local_time)
    #                         contention_map = self.get_contention_extra_map()
    #                         layer = 0
    #                         num_nodes = len(self.profiling.layers[layer])
    #                         action = np.zeros((num_nodes, 2), dtype=int)
    #                         action[:, 0] = layer
    #                         flat_idx = 0
    #                         for n_idx in range(num_nodes):
    #                             action[n_idx, 1] = t.assignment[flat_idx + n_idx]
    #                         extra = 0.0
    #                         if np.any(action[:, 1] == 1):
    #                             cloud_nodes = np.where(action[:, 1] == 1)[0]
    #                             key = (layer, cloud_nodes[0])
    #                             extra = contention_map.get(key, 0.0)
    #                         duration, new_pending = self.compute_layer_duration(
    #                             layer, action, t.cloud_pending_ms, None, bw, rtt, extra
    #                         )
    #                         t.remaining_time = duration
    #                         t.cloud_pending_ms = new_pending
    #                         t.total_latency_s += duration
    #                 else:
    #                     # Move to next layer
    #                     next_layer = t.current_layer + 1
    #                     t.current_layer = next_layer
    #                     self._refresh_contention_from_threads(threads)
    #                     local_time = self.cumulative_time_seconds + t.time_offset
    #                     bw = self._get_bandwidth_MBps_at_global_time(local_time)
    #                     rtt = self._get_rtt_at_global_time(local_time)
    #                     contention_map = self.get_contention_extra_map()
    #                     layer = next_layer
    #                     num_nodes = len(self.profiling.layers[layer])
    #                     action = np.zeros((num_nodes, 2), dtype=int)
    #                     action[:, 0] = layer
    #                     flat_idx = sum(len(self.profiling.layers[l]) for l in range(layer))
    #                     for n_idx in range(num_nodes):
    #                         action[n_idx, 1] = t.assignment[flat_idx + n_idx]
    #                     prev_layer = layer - 1
    #                     prev_num = len(self.profiling.layers[prev_layer])
    #                     prev_flat = sum(len(self.profiling.layers[l]) for l in range(prev_layer))
    #                     prev_action_pattern = tuple(t.assignment[prev_flat + i] for i in range(prev_num))
    #                     extra = 0.0
    #                     if np.any(action[:, 1] == 1):
    #                         cloud_nodes = np.where(action[:, 1] == 1)[0]
    #                         key = (layer, cloud_nodes[0])
    #                         extra = contention_map.get(key, 0.0)
    #                     duration, new_pending = self.compute_layer_duration(
    #                         layer, action, t.cloud_pending_ms, prev_action_pattern, bw, rtt, extra
    #                     )
    #                     t.remaining_time = duration
    #                     t.cloud_pending_ms = new_pending
    #                     t.total_latency_s += duration

    #         # ---- Progress log every 10 episodes ----
    #         if total_episodes_completed % 10 == 0 and total_episodes_completed > 0:
    #             bw_real = self.get_current_bandwidth_mbps()
    #             rtt_real = self.get_current_rtt_ms()
    #             # cont_map = self.get_contention_extra_map()
    #             # llama_extra = cont_map.get()
    #             # yolo_extra = cont_map.get((0, 0), 0.0)
    #             # bart_extra = cont_map.get()
    #             key = (self.i, self.j, self.k)
    #             row = self.contention_map.get(key)

    #             if row is None:
    #                 llama_extra = 0.0
    #                 yolo_extra = 0.0
    #                 bart_extra = 0.0
    #             else:
    #                 llama_extra = row.get('llama_extra', 0.0)
    #                 yolo_extra = row.get('yolos_extra', 0.0)
    #                 bart_extra = row.get('bart_extra', 0.0)
    #             print(f"[PROGRESS] T={self.cumulative_time_seconds:.2f}s | "
    #                   f"BW={bw_real:.1f}Mbps RTT={rtt_real:.1f}ms | "
    #                   f"Contention ({self.i},{self.j},{self.k}) | "
    #                   f"Extras: L={llama_extra:.2f}ms, Y={yolo_extra:.2f}ms, B={bart_extra:.2f}ms| "
    #                   f"Episodes: {total_episodes_completed}/{num_threads*episodes_per_thread}")
    #             # Write to CSV
    #             csv_writer.writerow([
    #                 self.cumulative_time_seconds,
    #                 bw_real,
    #                 rtt_real,
    #                 self.i, self.j, self.k,
    #                 llama_extra,yolo_extra,bart_extra, 
    #                 total_episodes_completed,
    #                 num_threads * episodes_per_thread
    #             ])
    #             csv_file.flush()

    #     # ---- Close CSV ----
    #     csv_file.close()
    #     print(f"[FILE] Contention data saved to {csv_filename}")

    #     # ---- Final summary ----
    #     total_seconds = self.cumulative_time_seconds
    #     total_ms = total_seconds * 1000.0
    #     overall_avg_ms = total_ms / (num_threads * episodes_per_thread)

    #     print(f"\n[FINISH] All threads completed {episodes_per_thread} episodes.")
    #     print(f"Total simulation time: {total_seconds:.2f}s ({total_ms:.2f} ms)")
    #     print(f"Overall average latency per episode: {overall_avg_ms:.3f} ms")

    #     # Per-thread average latency
    #     print("\n[STATS] Per-thread average latencies (ms):")
    #     for t in threads:
    #         avg_ms = (t.total_latency_s / episodes_per_thread) * 1000.0
    #         print(f"  Thread {t.thread_id}: {avg_ms:.3f} ms")