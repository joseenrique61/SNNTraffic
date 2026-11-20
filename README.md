# SNNTraffic: Spiking Neural Networks for Adaptive Traffic Signal Control

This project implements and compares two Deep Reinforcement Learning (DRL) agents for adaptive traffic signal control in the SUMO simulator using a **2x2 grid network** (4 intersections):

- **ANN-DQN**: A baseline Deep Q-Network using a standard Artificial Neural Network
- **Spiking-DQN**: A proposed Spiking Neural Network-based DQN using Leaky Integrate-and-Fire (LIF) neurons

Both agents use a **centralized architecture**: observations from all 4 traffic signals are concatenated, and the agent outputs actions for all signals simultaneously. They are compared against a **Fixed-Time** baseline controller with staggered phase offsets.

## Project Structure

```
SNNTraffic/
├── src/
│   └── notebook.ipynb    # Main notebook with full implementation
├── README.md
```

## Requirements

- Python 3.7+
- [SUMO](https://eclipse.dev/sumo/) (Simulation of Urban MObility)
- Python packages:
  - `sumo-rl`
  - `gymnasium`
  - `torch`
  - `matplotlib`
  - `numpy`
  - `snntorch`

## Setup and Installation

### 1. Install SUMO

**On Ubuntu/Debian:**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools
```

**On other systems**, follow the [official SUMO installation guide](https://sumo.dlr.de/docs/Downloads.php).

### 2. Set up Python environment

```bash
# Create a virtual environment (optional but recommended)
cd src/
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip pip install -r requirements.txt
```

## How to Run the Notebook

1. **Launch Jupyter** from the project root directory:

   ```bash
   cd SNNTraffic
   jupyter notebook
   ```

2. **Open the notebook**: Navigate to `src/notebook.ipynb` and open it.

3. **Run all cells**: Select `Cell → Run All` from the menu. The notebook will execute the following steps in order:

   | Section | Description |
   |---------|-------------|
   | **Setup** | Sets environment variables |
   | **Imports & Config** | Loads libraries, sets random seeds, configures hyperparameters |
   | **Environment Helpers** | Initializes the SUMO 2x2 grid environment |
   | **ANN-DQN Agent** | Defines the classical neural network and replay buffer |
   | **DQN Agent Class** | Full DQN agent implementation with epsilon-greedy exploration |
   | **Fixed-Time Baseline** | Evaluates a staggered fixed-time controller (3 episodes) |
   | **Train ANN-DQN** | Trains the ANN-based agent for 40 episodes (~1 hour simulated each) |
   | **Evaluate ANN-DQN** | Tests the trained ANN agent without exploration |
   | **Spiking-DQN Agent** | Defines the SNN architecture using `snntorch` with LIF neurons |
   | **Train Spiking-DQN** | Trains the SNN-based agent for 40 episodes |
   | **Generalization Test** | Tests both agents on 5 unseen random seeds |
   | **Results Comparison** | Generates comparison plots and computes metrics |
   | **Latency Benchmark** | Measures CPU inference latency for both models |
   | **Summary** | Final comparison table |

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `episodes` | 40 | Number of training episodes (short; increase for better results) |
| `sim_seconds` | 3600 | Simulation seconds per episode (1 simulated hour) |
| `delta_time` | 5 | Agent decision interval (seconds) |
| `hidden_dim` | 128 | Hidden layer size for both networks |
| `batch_size` | 64 | Experience replay batch size |
| `gamma` | 0.99 | Discount factor |
| `lr` | 1e-3 | Learning rate |
| `snn_num_steps` | 8 | SNN temporal window (timesteps) |
| `snn_beta` | 0.9 | SNN membrane decay rate |

## Observation & Action Space

- **Observation** per traffic signal: 21 dimensions (one-hot phase, min green, 8 densities, 8 queues)
- **Concatenated observation**: 84 dimensions (4 signals × 21)
- **Action** per signal: discrete 0–3 (select green phase)
- **Total action space**: 16 (4 signals × 4 actions)

## Results

After training, the notebook produces:
- A **comparison plot** (`comparison_ann_vs_snn.png`) showing training metrics
- **Model weights** saved as `ann_dqn_model.pth` and `snn_dqn_model.pth`
- **Generalization test** results on unseen traffic patterns
- **Latency benchmark** comparing CPU inference speed

### Expected Performance

| Method | Avg Waiting Time | Computational Cost |
|--------|----------------:|:------------------|
| Fixed-Time | ~1301s | 0 Ops |
| ANN-DQN | ~26s | ~29K MACs |
| Spiking-DQN | ~53s | ~1.4K SynOps |

The Spiking-DQN achieves competitive performance while potentially enabling more energy-efficient deployment through event-driven spike-based computation.

## References

- [SUMO](https://eclipse.dev/sumo/) - Simulation of Urban MObility
- [sumo-rl](https://github.com/LucasAlegre/sumo-rl) - Reinforcement Learning framework for SUMO
- [snnTorch](https://snntorch.readthedocs.io/) - Spiking neural network library for PyTorch