#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

# Set up the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from profiling.initialize.initialize_agx_profiling import get_LLM_profiling_data
from simulator.contention_generator import CloudEdgeSimulator


def main():
    # Load profiling data
    # profiling = get_LLM_profiling_data(deadline=500, edge_devices=1)
    profiling = get_LLM_profiling_data()

    # Paths to traces
    bandwidth_csv = ROOT_DIR / "simulator/data/bw_data.csv"
    contention_csv = ROOT_DIR / "simulator/data/contention.csv"

    # TFT model paths – adjust if needed
    # base_model_dir = ROOT_DIR / "artifacts" / "Idle" / "models"
    # bw_model_dir = base_model_dir / "bandwidth_mbps"
    # rtt_model_dir = base_model_dir / "rtt_ms"

    # Create simulator
    sim = CloudEdgeSimulator(profiling, bandwidth_csv, contention_csv)

    # Load TFT models if they exist
    # try:
    #     sim.load_tft_models(bw_model_dir, rtt_model_dir)
    #     print("✅ TFT models loaded successfully.")
    #     use_tft = True
    # except FileNotFoundError as e:
    #     print(f"⚠️ Could not load TFT models: {e}")
    #     print("   Falling back to using real trace values for predictions.")
    #     use_tft = False

    # Run simulation
    sim.simulate_concurrent(
        num_threads=9,
        episodes_per_thread=10,
        use_tft=False
    )


if __name__ == "__main__":
    main()
