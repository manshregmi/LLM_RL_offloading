import numpy as np

class EosSimulator:
    def __init__(self, graph_index: str):
        self.isEos1 = False
        self.isEos2 = False
        self.graph_index = graph_index

        match self.graph_index:
            case 'graph3':
                self.eos1_range = [[70, 100], [130, 150], [220, 250]]
                self.eos2_range = [[70, 100], [130, 150]]
            case 'graph1':
                self.eos1_range = [[70, 100], [200, 300], [350, 400]]
                self.eos2_range = []
            case 'graph2':
                self.eos1_range = [[150, 200], [300, 400]]
                self.eos2_range = [[60, 100]]

    def _find_containing_range(self, layer: int, ranges: list):
        for i, (low, high) in enumerate(ranges):
            if low < layer < high:
                return i, [low, high]
        return None, None

    def simulate_eos(self, layer: int, total_layers: int):
        # Safety terminal at last layer
        if layer + 1 >= total_layers:
            next_layer = layer
            terminal = True
            # print(f"EOS1={self.isEos1}, EOS2={self.isEos2} | layer {layer} -> terminal (end of total layers)")
            return self.isEos1, self.isEos2, next_layer, terminal

        terminal = False
        next_layer = layer + 1

        # Find which ranges contain current layer
        idx1, range1 = self._find_containing_range(layer, self.eos1_range)
        idx2, range2 = self._find_containing_range(layer, self.eos2_range)

        # Update EOS flags (parallel)
        if range1 is not None:
            if np.random.rand() < 0.075:
                self.isEos1 = True
                # print(f"EOS1 set to True at layer {layer} (range {range1})")
        else:
            if self.isEos1:
                # print(f"EOS1 reset at layer {layer} (outside all EOS1 ranges)")
                self.isEos1 = False

        if range2 is not None:
            if np.random.rand() < 0.075:
                self.isEos2 = True
                # print(f"EOS2 set to True at layer {layer} (range {range2})")
        else:
            if self.isEos2:
                # print(f"EOS2 reset at layer {layer} (outside all EOS2 ranges)")
                self.isEos2 = False

        # Prepare data for jump
        eos1_upper = range1[1] if range1 else None
        eos2_upper = range2[1] if range2 else None

        eos2_upper_for_idx1 = None
        if idx1 is not None and idx1 < len(self.eos2_range):
            eos2_upper_for_idx1 = self.eos2_range[idx1][1]

        eos1_upper_for_idx2 = None
        if idx2 is not None and idx2 < len(self.eos1_range):
            eos1_upper_for_idx2 = self.eos1_range[idx2][1]

        # Jump condition
        jump_triggered = False
        jump_target = None
        triggering = None

        if self.isEos1 and self.isEos2:
            if eos1_upper and eos2_upper:
                jump_target = min(eos1_upper, eos2_upper)   # smallest upper bound
            else:
                jump_target = eos1_upper or eos2_upper
            triggering = 'both'
            jump_triggered = True
        elif self.isEos1:
            should_jump = False
            if idx1 is not None:
                if idx1 >= len(self.eos2_range):
                    # No matching EOS2 range – jump, but terminal only if last index
                    should_jump = True
                elif eos2_upper_for_idx1 is not None and layer > eos2_upper_for_idx1:
                    should_jump = True
            if should_jump:
                jump_target = eos1_upper
                triggering = 'eos1'
                jump_triggered = True
        elif self.isEos2:
            should_jump = False
            if idx2 is not None:
                if idx2 >= len(self.eos1_range):
                    should_jump = True
                elif eos1_upper_for_idx2 is not None and layer > eos1_upper_for_idx2:
                    should_jump = True
            if should_jump:
                jump_target = eos2_upper
                triggering = 'eos2'
                jump_triggered = True

        if jump_triggered:
            next_layer = jump_target

            # TERMINAL ONLY IF JUMP COMES FROM THE LAST INDEX OF EOS1_RANGE
            terminal = False
            if triggering in ('eos1', 'both') and idx1 is not None:
                if idx1 == len(self.eos1_range) - 1:
                    terminal = True
            if triggering == 'eos2' and not terminal:   # if only eos2 triggered, check last eos2? But user said final index in eos1_range, so ignore eos2 for terminal
                # Optionally, you can also set terminal when eos2 is last and no eos1 match? 
                # But per instruction, only final eos1_range index triggers terminal.
                # So we leave terminal False for eos2-only jumps.
                pass

            # Reset flags
            self.isEos1 = False
            self.isEos2 = False
            # print(f"Jump triggered ({triggering}) from layer {layer} to {next_layer} | terminal={terminal}")
        # else:
            # print(f"No jump condition met at layer {layer} (EOS1={self.isEos1}, EOS2={self.isEos2})")

        # print(f"EOS1={self.isEos1}, EOS2={self.isEos2} | layer {layer} -> next_layer {next_layer}, terminal {terminal}")
        return self.isEos1, self.isEos2, next_layer, terminal