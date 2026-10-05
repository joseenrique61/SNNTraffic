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

Traffic congestion is a pervasive problem in modern urban environments, resulting in billions of hours of lost productivity, excessive fuel consumption, and increased greenhouse gas emissions @simeone2026modern. Traditional approaches to traffic light control include fixed-timed controllers, which operate on predetermined schedules, and actuated controllers, which use inductive loop detectors to extend green phases on demand. While simple and reliable, these methods fail to respond to complex, dynamic traffic patterns across multiple intersections.

Recent advances in Artificial Intelligence have produced Deep Reinforcement Learning (DRL) agents that learn adaptive traffic signal policies directly from simulation. These approaches, typically implemented with standard Artificial Neural Networks (ANNs), achieve superior performance in metrics such as average waiting time and intersection throughput. However, ANN-based controllers require continuous forward passes at every decision step, consuming considerable energy and computational resources. This makes them impractical for large-scale deployment on low-power edge devices at intersections.

Spiking Neural Networks (SNNs) offer a promising alternative. Unlike ANNs, which process information through continuous-valued activations, SNNs communicate via discrete, asynchronous binary spikes over time. This event-driven computation means that neurons only consume energy when they fire, making SNNs inherently more energy-efficient, particularly on neuromorphic hardware @simeone2026modern. Furthermore, SNNs naturally integrate temporal dynamics through their membrane potential, which can be beneficial for processing time-series data such as traffic states.

Our project addresses the following research question: Can a Spiking Deep Q-Network (Spiking-DQN) achieve competitive traffic signal control performance compared to conventional ANN-based DRL agents, while offering meaningful reductions in computational energy consumption?

This paper presents the proposal and preliminary framework for SNNTraffic, an SNN-based traffic light controller implemented in the SUMO simulation environment via the SUMO-RL interface. The remainder of this document is organized as follows. Section~2 reviews the related work on DRL for traffic control, SNN-based control, and neuromorphic computing. Section~3 describes the proposed methodology, including the SNN architecture, the reinforcement learning formulation, and the evaluation metrics. Section~4 presents a detailed 10-week work plan. Section~5 discusses risks and alternative strategies. Finally, Section~6 concludes with an AI usage declaration.

= Related Work

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

= Proposed Methodology

== Spiking Neuron Model

We adopt the Leaky Integrate-and-Fire (LIF) neuron model, the most widely used spiking neuron model in modern SNN research @simeone2026modern. The LIF dynamics are governed by:

$ V_m(t) = V_m(t-1) + (1 / tau) (I(t) - (V_m(t-1) - V_"rest")) $

where $V_m(t)$ is the membrane potential at time step $t$, $I(t)$ is the input current, $tau$ is the membrane time constant (equivalently controlled by the leak rate), and $V_"rest"$ is the resting potential. When the membrane potential exceeds a threshold $V_"th"$, the neuron emits a spike and resets to $V_"rest"$.

== Network Architecture

The proposed SNN consists of four layers:
- Input layer: 8 neurons, each encoding a feature of the traffic state (queue lengths and waiting times for each lane of the intersection).
- Hidden layers: two fully connected layers with 64 LIF neurons each.
- Output layer: $n_"actions"$ LIF neurons, each corresponding to a possible traffic light phase (e.g., North-South green, East-West green, etc.). The action is selected as the output neuron with the highest firing rate over the simulation time window.

All connections are fully connected between consecutive layers. The architecture is designed to be lightweight enough for real-time inference while retaining sufficient capacity to learn complex traffic patterns.

Key hyperparameters (tentative): time steps = 5--10, leak rate = 0.9, firing threshold = 1.0.

== Reinforcement Learning Formulation

We formulate traffic signal control as a Markov Decision Process (MDP) and solve it using a Deep Q-Network (DQN) architecture where the SNN acts as the Q-value approximator.

- State space: For each lane approaching the intersection, the state includes the queue length (number of vehicles waiting) and the cumulative waiting time. This yields a state vector of length $2 times n_"lanes"$.
- Action space: Discrete actions corresponding to traffic light phases (e.g., green for North-South direction, green for East-West direction). The number of actions depends on the intersection configuration.
- Reward function: The negative sum of waiting times across all vehicles in the simulation at each step $r_t = -sum_{v in V} w_v(t)$, where $w_v(t)$ is the waiting time of vehicle $v$ at time $t$. This incentivizes the agent to minimize overall delay.
- Training algorithm: We use a surrogate gradient-based approach (specifically, the super-spike surrogate gradient) to enable backpropagation through the spiking neurons during DQN training.

== Simulation Environment

We use SUMO (Simulation of Urban MObility) as the traffic simulator, interfaced through the SUMO-RL library. The simulation runs on Google Colab, which provides free GPU access for training while being a reproducible and accessible platform. The use of Colab represents a deliberate trade-off: it ensures easy reproducibility and zero-cost access, but imposes limits on available RAM, GPU time, and maximum session duration.

== Evaluation Metrics

We evaluate performance using the following metrics:
- Average waiting time per vehicle (primary metric)
- Average queue length at intersections
- Throughput (vehicles served per unit time)
- Estimated energy consumption per inference step (based on number of spikes fired, compared to equivalent ANN operations)

= Work Plan

The remaining 10 weeks are organized as descibred in #ref(<work_plan>).

#figure(
  caption: [Work plan for the remaining 10 weeks],
  table(
    columns: (1cm, auto, auto, auto),
    align: (left, left, left, left),
    inset: (x: 6pt, y: 3pt),
    stroke: 0.5pt,
    fill: rgb("#efefef"),

    [*Week*], [*Milestone*], [*Ariel*], [*José*],
    [6], [Setup SUMO-RL + Colab environment], [Environment setup], [Environment setup],
    [7], [Implement baseline DQN (ANN) agent], [Baseline implementation], [Literature review on SNN-RL],
    [8], [Implement SNN architecture], [Core SNN implementation], [Core SNN implementation],
    [9], [Integrate SNN with DQN], [SNN-RL training loop], [Integration with SUMO-RL],
    [10], [First experiments + debugging], [Experimentation], [Error analysis and debugging],
    [11], [Preliminary results + Deliverable 2], [Result analysis], [Report writing],
    [12], [Hyperparameter tuning], [Experiments], [Experiments],
    [13], [Comparative evaluation (SNN vs ANN)], [Analysis], [Analysis],
    [14], [Final experiments + ablation studies], [Experimentation], [Experimentation],
    [15], [Deliverable 3 report + final code], [Report writing], [Code documentation],
  )
) <work_plan>

= Risks and Alternatives

== Computational Limitations
Google Colab imposes session timeouts and limited GPU memory. *Alternative:* Reduce network size or simulation time steps; migrate to a local machine or cloud instance with higher limits if necessary.

== Convergence Difficulties
SNNs trained with surrogate gradients can be harder to converge than standard ANNs. *Alternative:* Use a hybrid approach where the policy is first learned with a standard ANN and then converted to an SNN, as demonstrated by Patel et al. @patel2019improved.

== Lack of Neuromorphic Hardware
We cannot guarantee access to neuromorphic chips such as Loihi or SpiNNaker2 for energy measurement. *Alternative:* Estimate energy consumption analytically by counting spike events and comparing with equivalent ANN FLOPs, following the methodology of Arfa et al. @arfa2025hardware.

== Training Instability
DQN with function approximation is known to suffer from instability, which may be exacerbated by the non-differentiable nature of spikes. *Alternative:* Implement experience replay and target networks; if issues persist, adopt a soft actor-critic framework as in SANSAC @arfa2025hardware.

== Simulation Overhead
SUMO itself consumes resources, and running it within Colab through SUMO-RL may introduce bottlenecks. *Alternative:* Reduce simulation speed.

= AI Usage Declaration

In accordance with academic integrity guidelines, we declare that portions of this document were drafted with assistance from an AI coding assistant (Opencode - DeepSeek V4). The AI was used to help structure the document, generate initial drafts of sections, and format the bibliography. All content was reviewed, edited, and approved by the authors, who take full responsibility for the accuracy and originality of the work.
