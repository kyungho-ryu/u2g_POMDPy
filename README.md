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


## Urban environment 
To realize multi-UAV environment of this study, we use the SUMO (simulation of urban mobility) simulator that is an open source simulator for urban vehicular and pedestrian traffic. We import a part of Berlin map, a 2×2 $km^2$ Mahattan grid and divide this environment region into 25 serving cells of 400×400 $m2$ like in figure considering UAV A2G coverage as an aerial base station. 
  

<img src="https://user-images.githubusercontent.com/73271891/234551258-12ab758f-aae0-45f1-a0a8-c1ceb273bd14.jpg" width="44%"/><img align="right" src="https://user-images.githubusercontent.com/73271891/234551083-da5a95a8-7bf9-4733-81c0-74b460f517bc.jpg" width="44%"/>


