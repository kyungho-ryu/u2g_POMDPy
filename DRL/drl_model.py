import numpy as np
from gym import spaces
from a2c import ActorCritic
import torch.optim as optim
import torch, logging

# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_steps = 60000
PRINT_INTERVAL = update_interval * 100

class DRLModel :
    def __init__(self):
        self.logger = logging.getLogger('POMDPy.DRLModel')
        self.logger.setLevel("INFO")
        self.state_dim = 50
        self.state_high = np.array([np.float32(24)] * self.state_dim)
        self.state_low = np.float32(np.zeros([self.state_dim]))
        self.state_space = spaces.Box(self.state_low, self.state_high, dtype=np.float32)

        self.action_dim = 25
        self.action_high = np.array([np.float32(24)] * self.action_dim)
        self.action_low = np.array([np.float32(0)] * self.action_dim)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        self.logger.info("State space : {}".format(self.state_space.shape[0]))
        self.logger.info("action space : {}".format(self.action_space.shape[0]))

        self.net_A2C = ActorCritic(self.state_space.shape[0], self.action_space.shape[0])
        self.optimizer = optim.Adam(self.net_A2C.parameters(), lr=learning_rate)

    def get_action(self, state):
        # state = np.array(np.arange(0, 25, 0.5))
        state = torch.from_numpy(state).float()
        mu_v = self.net_A2C.pi(state)
        mu = mu_v.data.cpu().numpy()
        mu = self.scale_action(mu)
        logstd = self.net_A2C.logstd.data.cpu().numpy()

        action = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        action = np.round(action).astype(int)
        action = np.clip(action, self.action_low[0], self.action_high[0])

        return action


    def scale_action(self, x) :
        # return (2/(1+np.exp(-2*x))) -1
        # return np.exp(3*x)
        return (self.action_high[0] * x + self.action_high[0]) / 2

