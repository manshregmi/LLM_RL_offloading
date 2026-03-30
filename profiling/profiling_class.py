import numpy as np


class ProfilingData:
    def __init__(
        self,
        numberOfEdgeDevice,
        layers,  # List of lists with node sizes per layer
        node_edge_times,  # Dict {(layer_idx, node_idx): comp_time on edge}
        node_cloud_times, # Dict {(layer_idx, node_idx): comp_time on cloud}
        bandwidth,
        rtt,
        input_size,       # Dict {(layer_idx, node_idx): output size in KB}
        node_edge_powers,  # Dict {(layer_idx, node_idx): power on edge}
        edge_idle_power,
        deadline,
        edge_communication_power,
        dependencies,
    ):
        self.numberOfEdgeDevice = numberOfEdgeDevice
        self.layers = layers
        self.node_edge_times = node_edge_times
        self.node_cloud_times = node_cloud_times
        self.bandwidth = bandwidth
        self.rtt = rtt
        self.input_size = input_size
        self.node_edge_powers = node_edge_powers
        self.edge_idle_power = edge_idle_power
        self.deadline = deadline
        self.edge_communication_power = edge_communication_power
        self.dependencies = dependencies

    def get_num_nodes(self, layer_idx):
        return len(self.layers[layer_idx])

    def get_node_edge_time(self, layer_idx, node_idx):
        return self.node_edge_times.get((layer_idx, node_idx), 0.0)

    def get_node_cloud_time(self, layer_idx, node_idx):
        return self.node_cloud_times.get((layer_idx, node_idx), 0.0)

    def get_node_edge_power(self, layer_idx, node_idx):
        return self.node_edge_powers.get((layer_idx, node_idx), 0.0)

    def get_total_nodes(self):
        total_nodes = sum(len(layer) for layer in self.layers)
        return total_nodes
    
    def get_total_edge_time(self):
        """Return total computation time if all nodes run on edge."""
        total_time = 0.0
        for layer_idx, layer in enumerate(self.layers):
            for node_idx in range(len(layer)):
                total_time += self.get_node_edge_time(layer_idx, node_idx)
        return total_time

    def get_edge_time_for_layer(self, layer_idx: int):
        """Return total edge time for all nodes in a given layer."""
        layer_time = 0.0
        for node_idx in range(len(self.layers[layer_idx])):
            layer_time += self.get_node_edge_time(layer_idx, node_idx)
        return layer_time

    def get_layer_total_edge_power(self, layer_idx):

        total_power = 0.0
        num_nodes = self.get_num_nodes(layer_idx)
        for node_idx in range(num_nodes):
            total_power += self.get_node_edge_power(layer_idx, node_idx)
        return total_power

    def get_layer_total_edge_time(self, layer_idx):
        """
        Return total computation time (ms) for all nodes in a layer.
        """
        total_time = 0.0
        num_nodes = self.get_num_nodes(layer_idx)
        for node_idx in range(num_nodes):
            total_time += self.get_node_edge_time(layer_idx, node_idx)
        return total_time

    def get_max_nodes(self):
        return max(len(layer) for layer in self.layers)
    
    def get_output_size(self, layer_idx, node_idx):
        return self.input_size.get((layer_idx, node_idx), 1.0)
    
    def get_max_layer_cloud_time(self, layer_idx):
        max_time = 0.0
        num_nodes = self.get_num_nodes(layer_idx)
        for node_idx in range(num_nodes):
            cloud_time = self.get_node_cloud_time(layer_idx, node_idx)
            if cloud_time > max_time:
                max_time = cloud_time
        return max_time
    
    def get_input_size(self):
        # Assuming input size is the output size of the first layer's first node
        return self.get_output_size(0, 0)
    
    def get_optimum_action_array(self):
        return [
            np.array([[0, 0]]),
            np.array([[1, 1]]),
            np.array([[2, 1]]),
            np.array([[3, 1], [3, 1], [3, 1]]),
            np.array([[4, 1], [4, 1], [4, 1]]),
            np.array([[5, 1], [5, 1], [5, 1]]),
            np.array([[6, 0]])
        ]
