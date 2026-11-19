#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [
    SNNTraffic: Application of Spiking Neural Networks on Traffic Lights, a Computationally Efficient Alternative to Common ANNs
  ],
  abstract: [
    Urban traffic congestion remains a critical challenge that directly impacts economic productivity, fuel consumption, and environmental pollution. Traditional traffic light controllers, such as fixed-timed and actuated systems, fail to adapt to dynamic traffic patterns, while recent Deep Reinforcement Learning (DRL) approaches based on standard Artificial Neural Networks (ANNs) incur high computational costs unsuitable for real-time and edge deployment. This paper proposes SNNTraffic, a Spiking Deep Q-Network (Spiking-DQN) for adaptive traffic signal control in the SUMO (Simulation of Urban MObility) environment. By leveraging the event-driven, neuromorphic nature of SNNs, our approach aims to achieve competitive traffic flow efficiency while significantly reducing computational energy consumption compared to conventional ANN-based controllers.
  ],
  authors: (
    (
      name: "Ariel Anchapaxi",
      department: [Software Engineering Student],
      organization: [Universidad de las Américas],
      location: [Quito, Ecuador],
      email: "ariel.anchapaxi@udla.edu.ec"
    ),
    (
      name: "José Cerezo",
      department: [Software Engineering Student],
      organization: [Universidad de las Américas],
      location: [Quito, Ecuador],
      email: "jose.cerezo@udla.edu.ec"
    ),
  ),
  index-terms: ("Spiking Neural Networks", "Reinforcement Learning", "Traffic Signal Control", "SUMO", "Deep Q-Network", "Neuromorphic Computing"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Traffic congestion is a pervasive problem in modern urban environments, resulting in billions of hours of lost productivity, excessive fuel consumption, and increased greenhouse gas emissions. Traditional approaches to traffic light control include fixed-timed controllers, which operate on predetermined schedules, and actuated controllers, which use inductive loop detectors to extend green phases on demand. While simple and reliable, these methods fail to respond to complex, dynamic traffic patterns across multiple intersections.

Recent advances in Artificial Intelligence have produced Deep Reinforcement Learning (DRL) agents that learn adaptive traffic signal policies directly from simulation. These approaches, typically implemented with standard Artificial Neural Networks (ANNs), achieve superior performance in metrics such as average waiting time and intersection throughput. However, ANN-based controllers require continuous forward passes at every decision step, consuming considerable energy and computational resources. This makes them impractical for large-scale deployment on low-power edge devices at intersections.

Spiking Neural Networks (SNNs) offer a promising alternative. Unlike ANNs, which process information through continuous-valued activations, SNNs communicate via discrete, asynchronous binary spikes over time. This event-driven computation means that neurons only consume energy when they fire, making SNNs inherently more energy-efficient, particularly on neuromorphic hardware @simeone2026modern. Furthermore, SNNs naturally integrate temporal dynamics through their membrane potential, which can be beneficial for processing time-series data such as traffic states.

Our project addresses the following research question: Can a Spiking Deep Q-Network (Spiking-DQN) achieve competitive traffic signal control performance compared to conventional ANN-based DRL agents, while offering meaningful reductions in computational energy consumption?

This paper presents SNNTraffic, an SNN-based traffic light controller implemented in the SUMO simulation environment via the SUMO-RL interface. The remainder of this document is organized as follows. @related reviews the related work on DRL for traffic control, SNN-based control, and neuromorphic computing. @methods describes the full methodology, including the simulation environment, RL formulation, and network architectures for both the ANN-DQN baseline and the proposed Spiking-DQN. @prel_results presents preliminary experimental results with training curves, generalization metrics, and computational efficiency analysis. @prel_disc discusses the implications of our findings and current limitations. @adj_work_plan presents an updated work plan for the remaining 5 weeks. @risks discusses remaining risks and alternative strategies. Finally, @ai_declaration concludes with an AI usage declaration.

= Related Work <related>

== Deep Reinforcement Learning for Traffic Signal Control

Deep Reinforcement Learning has been widely applied to adaptive traffic signal control in recent years. Han et al. @han2024cyclight proposed CycLight, a cycle-level DRL approach that uses a Parameterized Deep Q-Network (DQN) to optimize both cycle length and phase splits simultaneously in the SUMO simulator. Their approach demonstrates that network-level coordination can significantly reduce average travel time. Similarly, Wang and Wang proposed Friend-DQN, a multi-agent DRL framework in which agents cooperate during training to reduce the state-action space, achieving faster convergence on SUMO-based traffic networks. These works establish strong DRL baselines that our SNN-based approach aims to match or approach while reducing energy consumption.

== Spiking Neural Networks for Control Tasks

The application of SNNs to control tasks has gained traction as training methods have matured. Bing et al. @bing2020indirect investigated both indirect training (converting a pre-trained ANN-based DQN into an SNN) and direct training (using reward-modulated STDP) for end-to-end lane-keeping vehicle control. Their results show that SNNs can maintain competitive control performance while benefiting from event-driven computation. Patel et al. @patel2019improved demonstrated that policies learned via a standard DQN can be converted to SNN platforms with improved robustness to occluded or noisy input, a significant advantage for real-world traffic scenarios where sensor noise is inevitable.

Recently, Ghoreishee et al. @ghoreishee2026fuzzy proposed a fuzzy encoding-decoding framework for deep spiking Q-networks in autonomous driving tasks. Their work addresses two key limitations of SNN-RL: information loss during dense-to-spike conversion and the limited representational capacity of spike-based value functions. Their results on the HighwayEnv benchmark substantially close the performance gap between spiking and non-spiking Q-networks.

== Energy-Efficient Neuromorphic Deployment

Arfa et al. @arfa2025hardware demonstrated an energy-efficient implementation of spiking Q-networks on the SpiNNaker2 neuromorphic platform for control tasks, achieving up to 32x energy reduction compared to GPU baselines. Their work provides strong evidence that SNN-RL controllers can be practically deployed on low-power hardware. Chen and Spanogianopoulos @chen2023autonomous studied SNNs for traffic light change detection using Dynamic Vision Sensor (DVS) data in the CARLA simulator, demonstrating that SNNs can process event-based visual data for traffic perception with low latency.

== Foundational Neuromorphic Background

Simeone @simeone2026modern provides a comprehensive survey of modern neuromorphic AI, reviewing training methodologies including surrogate gradients and RL-based local learning rules. This survey situates SNNs within the broader landscape of efficient AI and provides theoretical grounding for the event-driven computation paradigm that underpins our approach.

== Research Gap

To the best of our knowledge, no existing work directly combines Spiking Neural Networks with Deep Q-Networks for adaptive traffic signal control in the SUMO simulation environment. The convergence of SNN-based control and traffic signal optimization represents a novel research direction with potential practical benefits in energy-efficient smart city infrastructure.

= Methods <methods>

== Simulation Environment

All experiments are conducted using the SUMO (Simulation of Urban MObility) traffic simulator interfaced through the SUMO-RL library @sumorl. We use the built-in _2x2 grid network_ provided by SUMO-RL, which consists of four signalized intersections in a $2 times 2$ layout. Each intersection has 8 incoming lanes and supports 4 green phases. The network is configured with a symmetric route file that generates traffic flows uniformly across all directions.

The agent makes a control decision every $Delta t = 5$ simulation seconds (the SUMO-RL `delta_time` parameter), and each episode runs for 3,600 simulation steps (1 simulated hour). By default, SUMO simulation performs best-effort real-time execution, meaning simulation speed adapts to available compute resources. All training runs were performed on a system with an Nvidia GPU with CUDA, while inference latency benchmarks were conducted on CPU to reflect realistic embedded deployment conditions.

=== Fixed-Time Baseline

We implement a staggered fixed-time controller as a naive baseline. Each traffic signal cycles through its 4 green phases independently. Each phase lasts $6 "steps" times 5 "s/step" = 30 "s"$, and intersections are phase-offset by 1 phase (i.e., TS 1 starts at phase 0, TS 2 at phase 1, etc.) to prevent all signals from turning green simultaneously. The fixed-time controller requires zero computational operations and represents the simplest possible traffic signal policy.

== Reinforcement Learning Formulation

We formulate adaptive traffic signal control across a multi-intersection grid as a Markov Decision Process (MDP). Both the ANN-DQN baseline and the proposed Spiking-DQN share the same MDP formulation.

=== State Space

In the 2x2 SUMO-RL grid, each traffic signal receives a per-agent observation vector with 21 dimensions:
- 4 one-hot encoded for the currently active green phase
- 1 binary `min_green` indicator
- 8 lane densities (vehicles per lane)
- 8 lane queue lengths (stopped vehicles per lane)

Because we use a _centralized architecture_, the full state is the concatenation of all 4 agents' observations, yielding a total observation dimension of $4 times 21 = 84$. This allows the agent to observe the global traffic state and learn coordinated policies across intersections.

=== Action Space

Each traffic signal selects among 4 discrete actions, corresponding to the 4 possible green phases. The centralized agent outputs 4 actions simultaneously, one per intersection. The total action dimension is $4 times 4 = 16$, but actions are reshaped and argmaxed per-agent during inference.

=== Reward Function

The reward at each time step $t$ is defined as the negative total system waiting time, normalized: $
r_t = - "system_total_waiting_time"_t / 1000
$ where "system_total_waiting_time" is the aggregated waiting time across all vehicles in the simulation, as reported by SUMO's `system_total_waiting_time` metric. This formulation incentivizes the agent to minimize the total accumulated delay across all vehicles and all intersections.

=== Training Algorithm

Both agents are trained using the Deep Q-Network (DQN) algorithm with experience replay and target networks. The key DQN hyperparameters are shared between both agents:

#figure(
  caption: [Shared DQN hyperparameters for both agents],
  table(
    columns: (auto, auto),
    align: (left, left),
    inset: (x: 6pt, y: 3pt),
    stroke: 0.5pt,
    fill: rgb("#efefef"),

    [*Parameter*], [*Value*],
    [Episodes], [40],
    [Simulation steps per episode], [3,600 (1 hour)],
    [Decision interval ($Delta t$)], [5 s],
    [Batch size], [64],
    [Discount factor ($gamma$)], [0.99],
    [Learning rate], [$10^-3$],
    [Initial epsilon], [1.0],
    [Final epsilon], [0.05],
    [Epsilon decay rate], [0.90 per episode],
    [Target network update frequency], [Every 5 episodes],
    [Replay buffer capacity], [10,000 transitions],
    [Optimizer], [Adam],
    [Loss function], [Huber (Smooth L1)],
  )
) <dqn_hyperparams>

== ANN-DQN Agent (Baseline)

The baseline agent uses a standard deep Artificial Neural Network as the Q-value approximator. The architecture consists of:

- _Input layer_: 84 neurons (concatenated observations from 4 intersections)
- _Hidden layer 1_: Fully connected layer with 128 neurons and ReLU activation
- _Hidden layer 2_: Fully connected layer with 128 neurons and ReLU activation
- _Output layer_: 16 neurons (4 actions per intersection $times$ 4 intersections)

The total computational cost of a single forward pass is calculated as the number of multiply-accumulate operations (MACs):

$"MACs" = (84 times 128) + (128 times 128) + (128 times 16) = 29,184$

The policy network and a separate target network (periodically synchronized) are both instances of this architecture.

== Spiking-DQN Agent (Proposal)

The proposed agent replaces the ANN with a Spiking Neural Network that uses Leaky Integrate-and-Fire (LIF) neurons. The architecture mirrors the ANN's width for a fair comparison:

- _Input layer_: 84 neurons (direct encoding: continuous-valued inputs are fed as analog currents)
- _Hidden layer 1_: 128 LIF neurons, fully connected from input
- _Hidden layer 2_: 128 LIF neurons, fully connected
- _Output layer_: 16 LIF integrator neurons (no reset, membrane potential accumulates over time)

=== Spiking Neuron Dynamics

We use the Leaky Integrate-and-Fire (LIF) model as implemented in the snnTorch library @eshraghian2021training. The membrane potential $V_m$ evolves over discrete time steps as: $
V_m(t) = beta V_m(t-1) + I(t)
$ where $beta = 0.9$ is the membrane decay constant (equivalent to leak rate), and $I(t)$ is the input current at time $t$. When $V_m$ exceeds the firing threshold $theta = 1.0$, the neuron emits a binary spike and resets $V_m$ to 0. The output layer uses `reset_mechanism = "none"`, allowing it to integrate incoming spikes over the full temporal window without resets.

=== Temporal Processing

Unlike an ANN that produces an output in a single forward pass, the SNN operates over a temporal window of $T = 8$ time steps. For each state observation:

1. The same input $x$ is injected as a constant current to the input layer for $T$ steps.
2. At each step $t$, spikes propagate through the two hidden layers, and the output layer integrates its membrane potential.
3. After $T$ steps, the final Q-value estimate is the temporal average of the output layer's membrane potential over the window: $
Q(s, a) = 1 / T sum_(t=1)^T V_m^(o u t)(t)
$ where $V_m^(o u t)(t)$ is the membrane potential of the output layer at step $t$.

=== Surrogate Gradient

To enable backpropagation through the non-differentiable spike function, we use the fast sigmoid surrogate gradient implemented in snnTorch with a slope parameter of 25: $
(partial s p k) / (partial V) &approx sigma'(V) \
&= 1 / (1 + exp(-25 V))^2 dot 25 dot exp(-25 V)
$

=== Computational Cost

The computational cost of an SNN is measured in Synaptic Operations (SynOps), which count only the spikes that actually propagate through the network. Given an average spike rate per inference: $
"SynOps" = "avg spikes per step" / T times "hidden dim"
$ 

Using the empirical average of approximately 88 spikes per inference step across the last 5 training episodes: $
"SynOps" approx 88 / 8 times 128 approx 1,429
$

This represents a _\~20x reduction_ in computational operations compared to the ANN's 29,184 MACs, highlighting the energy efficiency potential of event-driven computation.

=== Training Protocol

Both agents follow identical training protocols to ensure a fair comparison:
1. Each episode begins with `env.reset(seed = SEED + 1000 + ep)` to provide the same traffic pattern sequence to both agents.
2. At each decision step (every 5 simulation seconds), the agent selects actions using epsilon-greedy exploration.
3. The transition `(state, actions, reward, next_state, done)` is stored in the replay buffer.
4. A minibatch of 64 transitions is sampled for gradient descent using the Huber loss against target Q-values.
5. After each episode, epsilon decays: $epsilon <- max(0.05, epsilon times 0.90)$.
6. Every 5 episodes, the target network copies the policy network weights.
7. Training runs for 40 episodes total.

== Evaluation Metrics

We evaluate performance across four dimensions:

- *Average waiting time per step* (primary traffic efficiency metric): The mean of `system_total_waiting_time` across all timesteps in an episode.
- *Average queue length*: The mean number of stopped vehicles across all timesteps.
- *Computational cost*: MACs for the ANN vs. SynOps for the SNN, as a proxy for energy consumption.
- *CPU inference latency*: Wall-clock time per forward pass on CPU, measured over 1,000 iterations after 50 warm-up iterations.

= Preliminary Results <prel_results>

We present the experimental results comparing the Fixed-Time baseline, the ANN-DQN agent, and the proposed Spiking-DQN agent on the 2x2 grid traffic network.

== Summary of Results

#figure(
  caption: [Summary of results comparing Fixed-Time, ANN-DQN, and Spiking-DQN. Waiting times and queue lengths are averaged over the final trained episode (evaluation with zero exploration). Generalization results are mean $plus.minus$ std over 5 unseen seeds.],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    inset: (x: 6pt, y: 3pt),
    stroke: 0.5pt,
    fill: (rgb("#efefef"), none),

    [*Metric*], [*Fixed-Time*], [*ANN-DQN*], [*Spiking-DQN*],
    [Avg waiting time (s)], [1,301.03], [25.94], [53.33],
    [Avg queue (vehicles)], [--], [14.69], [24.60],
    [Generalization wait (s)], [--], [$16.71 plus.minus 1.76$], [$22.62 plus.minus 1.84$],
    [Generalization queue], [--], [$2.41 plus.minus 0.12$], [$3.11 plus.minus 0.06$],
    [Computational cost], [0 Ops], [29,184 MACs], [\~1,429 SynOps],
    [CPU inference latency], [N/A], [0.0164 ms], [1.3250 ms],
  ),
) <results_table>

== Traffic Efficiency

Both DRL agents dramatically outperform the fixed-time controller. The Fixed-Time baseline produces an average waiting time of 1,301.03 s per step, which is expected since phase timings are static and cannot adapt to fluctuating traffic demand.

The ANN-DQN agent achieves the best traffic efficiency, reaching a final-episode average waiting time of _25.94 s_ per step—a _98.0% reduction_ compared to Fixed-Time. The Spiking-DQN achieves _53.33 s_, representing a _95.9% reduction_ from Fixed-Time. The ANN-DQN outperforms the Spiking-DQN by approximately $2.1 times$ in waiting time on the training seed.

=== Generalization Performance

On 5 unseen traffic seeds (seeds 2000-2004), the ANN-DQN maintains a mean waiting time of $16.71 plus.minus 1.76$ s, while the Spiking-DQN achieves $22.62 plus.minus 1.84$ s. Several observations are noteworthy:

1. Both agents generalize well to unseen traffic patterns, with low standard deviations ($plus.minus 1.76$ and $plus.minus 1.84$ s respectively).
2. The generalization performance is actually better than the final training episode performance for both agents. This is likely because the held-out seeds produce slightly less congested traffic patterns than the training seed sequence.
3. The gap between ANN-DQN and Spiking-DQN narrows in generalization ($22.62 / 16.71 approx 1.35 times$) compared to the training seed ($53.33 / 25.94 approx 2.06 times$), suggesting the SNN may generalize relatively better to easier traffic conditions.

==== Training Curves

#figure(
  caption: [Training curves comparing ANN-DQN and Spiking-DQN across 40 episodes. Top row: global average waiting time and queue size. Middle row: per-intersection averages. Bottom row: loss and reward evolution.],
  image("src/comparison_ann_vs_snn.png", width: 100%),
) <training_curves>

@training_curves shows the complete training progression. Key observations:

- _Convergence speed_: The ANN-DQN converges to its stable regime within approximately 28 episodes, while the Spiking-DQN requires \~35 episodes to reach comparable stability. This is expected given the additional complexity of surrogate gradient-based SNN optimization.
- _Loss dynamics_: Both agents show smooth loss reduction. The SNN loss starts higher (0.25 vs. 0.08) but converges to a similar floor (approx 0.01).
- _Spike sparsity_: The Spiking-DQN maintains a consistent spike rate of approximately 85-92 spikes per inference step throughout training, confirming stable spiking dynamics.

== Computational Efficiency

=== Synaptic Operations vs. MACs

The theoretical computational cost differs substantially between the two architectures:

- _ANN-DQN_: 29,184 MACs per forward pass (all weights compute regardless of input)
- _Spiking-DQN_: \~1,429 SynOps per inference (only active synapses corresponding to emitted spikes)

This represents an approximate _20x reduction_ in computational operations for the SNN. On neuromorphic hardware where each spike consumes only a picojoule of energy, this translates to near-proportional energy savings. On conventional hardware (CPU/GPU), however, the efficiency advantage is not directly realized because dense matrix operations are executed regardless of spike sparsity.

=== CPU Inference Latency

We benchmarked single-inference latency on CPU (1000 iterations after warm-up):

- _ANN-DQN_: 0.0164 ms per inference
- _Spiking-DQN_: 1.3250 ms per inference

The SNN is approximately _80x slower_ on CPU. This is expected because the SNN must unroll $T = 8$ time steps of sequential computation per inference, and the current PyTorch implementation executes dense matrix multiplications at each step regardless of sparsity. On dedicated neuromorphic hardware (e.g., Loihi, SpiNNaker2), this latency overhead is eliminated because event-driven processing naturally exploits spike sparsity @arfa2025hardware.

Despite this, the SNN latency of 1.33 ms is well within the real-time constraint of a 5-second decision interval, confirming feasibility for deployment even on CPU.

= Preliminary Discussion <prel_disc>

== Interpretation of Results

The experimental results demonstrate a clear performance-efficiency trade-off between ANN-DQN and Spiking-DQN architectures for adaptive traffic signal control.

=== Accuracy Gap

The Spiking-DQN achieves competitive traffic performance (53.33 s waiting time vs. 25.94 s for ANN-DQN), but does not fully close the gap. The 2x performance difference can be attributed to several factors:

1. _Temporal averaging approximation_: Using the temporal average of the output membrane potential as Q-values introduces a loss of temporal precision. The state encoding is static (identical input for all $T$ timesteps), yet the SNN's temporal dynamics may introduce spurious variation in the output trajectory that the averaging smooths out.
2. _Surrogate gradient noise_: The fast sigmoid surrogate gradient provides only an approximation of the true gradient through the spike function. This noisy gradient signal makes optimization less efficient than the exact gradient computation in ANNs.
3. _Limited temporal window_: With $T = 8$, the SNN has limited time to propagate information through the two hidden layers. A larger $T$ might improve representational capacity but would incur higher latency.

=== Generalization Behavior

Both agents generalize well to unseen seeds. The SNN shows slightly higher variance ($plus.minus 1.84$ vs. $plus.minus 1.76$ s), which is consistent with the noisier optimization landscape of surrogate gradient training. An interesting finding is that both agents perform _better_ on the generalization seeds than on the final training seed, suggesting the training seed produces relatively high-congestion conditions.

=== Computational Trade-offs

The core motivation for the SNN approach is validated by the SynOps analysis: the Spiking-DQN requires approximately 20x fewer computational operations than the equivalent ANN-DQN during inference. This advantage would be fully realized on neuromorphic hardware, where energy consumption scales directly with spike activity. On conventional hardware, the SNN incurs a latency penalty due to sequential temporal unrolling, but still operates well within real-time constraints (1.33 ms vs. a 5,000 ms decision interval).

== Limitations

Several limitations should be acknowledged:

1. _Training duration_: With only 40 episodes of training, neither agent has fully converged. The ANN-DQN's waiting time was still decreasing in the final episodes, and the Spiking-DQN showed higher variance with a downward trend. More episodes may close the performance gap.
2. _Network complexity_: The 2x2 grid is a relatively simple topology. Real-world traffic networks involve dozens or hundreds of intersections with asymmetric traffic flows, turning movements, and pedestrian crossings. Scalability to larger networks remains to be tested.
3. _Direct encoding_: We use direct encoding (continuous-valued input as current), which bypasses the potential advantages of temporal encoding schemes used in biological sensory systems. Rate-based or temporal encoding might improve SNN performance.
4. _No neuromorphic hardware_: Energy estimates are analytical (SynOps) rather than measured directly on neuromorphic hardware. Actual energy savings depend on the specific hardware platform used.
5. _Single environment configuration_: Both the topology (2x2 grid) and traffic patterns (uniform demand) are fixed. Robustness across diverse networks and traffic demand scenarios has not been evaluated.

= Adjusted Work Plan <adj_work_plan>

Based on our preliminary results, we adjust the remaining 5 weeks of the project as follows.

#figure(
  caption: [Adjusted work plan for the remaining 5 weeks],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    inset: (x: 6pt, y: 3pt),
    stroke: 0.5pt,
    fill: rgb("#efefef"),

    [*Week*], [*Milestone*], [*Ariel*], [*José*],
    [11 (Done)], [Deliverable 2], [Result analysis], [Report writing],
    [12], [Hyperparameter tuning + extended training], [Grid search (lr, $beta$, $T$)], [Extended training (100+ episodes)],
    [13], [Comparative evaluation (SNN vs ANN)], [Statistical analysis], [Per-intersection breakdown],
    [14], [Ablation studies + scalability test], [Ablation (encoding, topology)], [Test on larger networks],
    [15], [Deliverable 3 report + final code], [Report writing], [Code documentation],
  )
) <work_plan>

= Risks and Alternatives <risks>

== Performance Gap
The Spiking-DQN achieves competitive but inferior traffic efficiency compared to the ANN-DQN (53.33 s vs. 25.94 s waiting time). *Alternative:* Increase the temporal window $T$, apply rate-based or temporal encoding schemes, or perform ANN-to-SNN conversion as demonstrated by Patel et al. @patel2019improved.

== Training Duration
With only 40 episodes, neither agent has fully converged. The SNN's higher variance and slower convergence suggest more episodes would benefit the spiking agent. *Alternative:* Scale training to 100-200 episodes and tighten epsilon decay for the SNN.

== Computational Overhead
The SNN is 80x slower on CPU due to sequential temporal unrolling. *Alternative:* This overhead is specific to conventional hardware; on neuromorphic platforms (Loihi, SpiNNaker2), the SNN would realize its SynOps advantage as actual energy savings. For CPU deployment, the ANN is the more practical choice.

== Lack of Neuromorphic Hardware
We cannot guarantee access to neuromorphic chips for real energy measurement. *Alternative:* Continue analytical estimation via SynOps, and consider cloud-based neuromorphic simulators or remote access to SpiNNaker2 if available.

= AI Usage Declaration <ai_declaration>

In accordance with academic integrity guidelines, we declare that portions of this document were drafted with assistance from an AI coding assistant (Opencode - DeepSeek V4). The AI was used to help structure the document, generate initial drafts of sections, and format the bibliography. All content was reviewed, edited, and approved by the authors, who take full responsibility for the accuracy and originality of the work.
