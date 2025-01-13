# Energy-Efficient Deployment of Aerial Base Stations for Mobile Users in Multi-Hop UAV Networks

## Overview
Unmanned Aerial Vehicles (UAVs) are increasingly deployed as aerial base stations in Low-Altitude Platforms (LAPs) to provide wireless connectivity to ground users, particularly in disaster-stricken areas or regions with limited digital infrastructure. While previous studies have focused on energy-efficient UAV deployment for stationary ground users, this project addresses deployment for mobile ground users (GMUs) who are partially observable in a multi-UAV network.

### Key Contributions:
- **Online Deployment Algorithm**: Proposes an online deployment strategy for GMUs using a Partially Observable Markov Decision Process (POMDP) framework.
- **Monte Carlo Tree Search (MCTS) Enhancements**: Incorporates Double Progressive Widening (DPW) and Deep Neural Network (DNN)-based guidance into the MCTS algorithm.
- **Dynamic Learning**: Employs Proximal Policy Optimization (PPO) for efficient and adaptive DNN training using simulated samples.
- **Dual Objective Optimization**: Balances energy savings and throughput, factoring in prediction errors in GMU locations.

From experiments in urban environments, the proposed approach demonstrates superior learning performance, faster convergence, and efficient trade-offs between energy and throughput compared to traditional methods.

---

## Urban Environment Simulation
To model a realistic multi-UAV environment, the **SUMO (Simulation of Urban Mobility)** simulator was used. 

### Simulation Details:
- **Map and Division**: A portion of the Berlin map was imported, representing a 2x2 km<sup>2</sup> Manhattan grid. The region is divided into 25 serving cells, each 400x400 m<sup>2</sup> (as shown in the figure below), corresponding to the A2G coverage area of UAVs.

<p align="center">
  <img src="https://user-images.githubusercontent.com/73271891/234551258-12ab758f-aae0-45f1-a0a8-c1ceb273bd14.jpg" width="44%"/>
  <img src="https://user-images.githubusercontent.com/73271891/234551083-da5a95a8-7bf9-4733-81c0-74b460f517bc.jpg" width="44%"/>
</p>

#### Key Features:
- **GMU Mobility**:
  - Represented by triangle symbols with direction indicators.
  - Move toward random destinations along roads at random speeds [0–1.4 m/s].
  - **[Trajectory Data](https://github.com/kyungho-ryu/u2g_POMDPy/tree/master/mobility/original_trajectory)**: Logged every 300s, providing sufficient time for GMUs to change cells.
- **UAV Deployment**:
  - UAVs are deployed at the center of each cell at a 20m altitude.
  - Adjacent UAVs (including diagonal ones) maintain A2A links.
- **Ground Control Center (GCC)**:
  - Controls UAV trajectories and locations via ad-hoc networking under partially observable conditions based on GMU mobility.
  - GMUs access the Internet through the multi-UAV network.

---

## GMU Mobility Prediction Model
A simulator was developed based on the **Cell-based Probabilistic Trajectory Prediction (CPTP)** model to predict GMU mobility. This model dynamically reconstructs itself using observed GMU trajectories, making it suitable for disaster environments with limited prior data.

### Model Features:
- Semi-lazy GMU mobility model.
- Continuously improves performance with new trajectory observations.

### Resources:
- **[CPTP Model Code](https://github.com/kyungho-ryu/u2g_POMDPy/blob/master/mobility/semi_lazy.py)**

### Performance:
<p align="center">
  <img src="https://user-images.githubusercontent.com/73271891/234559297-5c8501cb-9774-46bd-a55e-f69173729c1c.jpg" width="44%">
</p>

---

## MCTS Guided by Deep Reinforcement Learning
This study extends the **MCTS-based Partially Observable Monte Carlo Planning (POMCP)** framework to address partially observable environments. 

### Methodology:
- **Deep Neural Network (DNN) Guidance**: Accelerates the MCTS search by narrowing the action space.
- **Actor-Critic Network Training**:
  - Simulation samples are used for online training.
  - **Proximal Policy Optimization (PPO)** is applied to prevent overfitting from simulation samples.

### Performance:
<p align="center">
  <img src="https://user-images.githubusercontent.com/73271891/234560638-3c3f0377-a0a0-4a81-8672-eb6c8ad14976.png" width="88%">
</p>

#### Figure Details:
- **Color Index**: Indicates the number of actual GMUs in each cell.
- **Cell Numbers**: Represent the average number of GMUs sampled from the belief state.
- **Dashed Circles**: Indicate UAV locations as determined by the selected action.

---

## Repository Structure
```plaintext
├── mobility
│   ├── original_trajectory       # Original GMU trajectory data
│   ├── semi_lazy.py              # CPTP mobility prediction model
├── simulation
│   ├── sumo_simulation           # SUMO environment setup and scripts
│   ├── map_data                  # Imported Berlin map data
├── models
│   ├── pomcp                     # MCTS-based POMDP planning framework
│   ├── drl_guidance              # Deep reinforcement learning modules
└── README.md                     # Project documentation
```

---

## How to Run
1. **Set Up SUMO Environment**:
   - Install [SUMO](https://www.eclipse.org/sumo/).
   - Import the Berlin map data from the `simulation/map_data` directory.
2. **Run GMU Mobility Simulation**:
   ```bash
   python mobility/semi_lazy.py
   ```
3. **Train and Deploy UAV Control Algorithm**:
   - Train the DNN-guided MCTS:
   ```bash
   python models/drl_guidance/train.py
   ```
   - Deploy the trained model for UAV control:
   ```bash
   python models/pomcp/deploy.py
   ```

---

## References
- [POMCP: Partially Observable Monte Carlo Planning](https://proceedings.neurips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf)
- [SUMO: Simulation of Urban Mobility](https://www.eclipse.org/sumo/)
