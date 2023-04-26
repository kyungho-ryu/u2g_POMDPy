# Energy efficient deployment of aerial base stations for mobile users in multi-hop UAV networks

Unmanned aerial vehicles (UAVs) are popularly considered as aerial base stations in a Low-Altitude Platform (LAP) to provide
wireless connections to ground users in disaster and digital divide situation. Most of previous studies have investigated energy
efficient UAV deployment only for the stationary ground users. Contrarily, we propose an on-line deployment algorithm for ground
mobile users (GMUs) who are partially observable in the multi-UAV system. In the framework of Partially Observable Markov
Decision Process (POMDP) with large state and action space, we use the Monte Carlo Tree Search (MCTS) algorithm enhanced
by the Double Progressive Widening (DPW) and Deep Neural Network (DNN)-based guidance. For online learning of the DNN,
simulated samples are used with Proximal Policy Optimization (PPO) that prevents excessive training for a particular belief. From
experiment in urban environment, the proposed scheme shows better learning performance than the MCTS with rollout policy and
faster convergence than training with environment samples. In addition, it optimizes dual objectives proportionally in a trade-off
between energy saving and throughput including prediction error on GMU location.

