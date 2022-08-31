import numpy as np
from gym import spaces
from DRL.a2c import ActorCritic
from examples.u2g.u2g_action import U2GAction
import torch.optim as optim
import torch, logging, math
import torch.nn.functional as F

# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_steps = 60000
PRINT_INTERVAL = update_interval * 100

class DRLModel :
    def __init__(self, state_dim, state_space, action_dim, action_space):
        self.logger = logging.getLogger('POMDPy.DRLModel')
        self.logger.setLevel("INFO")
        self.state_dim = state_dim
        self.state_high = np.array([np.float32(state_space[1])] * self.state_dim)
        self.state_low = np.array([np.float32(state_space[0])] * self.state_dim)
        self.state_space = spaces.Box(self.state_low, self.state_high, dtype=np.float32)

        self.action_dim = action_dim
        self.action_high = np.array([np.float32(state_space[1])] * self.action_dim)
        self.action_low = np.array([np.float32(state_space[0])] * self.action_dim)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        self.logger.info("State dimension, space : {}, {}".format((self.state_low[0], self.state_high[0]), self.state_space.shape[0]))
        self.logger.info("action dimension, space : {}, {}".format((self.action_low[0], self.action_high[0]), self.action_space.shape[0]))

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
        action = np.clip(action, self.action_low[0], self.action_high[0])
        action = np.round(action).astype(int)

        return U2GAction(list(action))


    def scale_action(self, x) :
        # return (2/(1+np.exp(-2*x))) -1
        # return np.exp(3*x)
        return (self.action_high[0] * x + self.action_high[0]) / 2

    def get_value(self, state):
        state = torch.from_numpy(state).float()
        value = self.net_A2C.V(state)
        value = value.data.cpu().numpy()[0]

        return value

    def update(self, sample):
        td_target_vec = torch.from_numpy(np.array(sample.td_target)).float()
        s_vec = torch.from_numpy(np.array(sample.s_list)).float()
        a_vec = torch.from_numpy(np.array(sample.a_list)).float()
        value = self.net_A2C.V(s_vec).reshape(-1)
        print("td_target_vec", td_target_vec, td_target_vec.shape)
        print("value", value, value.shape)

        advantage = td_target_vec - value
        print("advantage", advantage, advantage.shape)

        print("a_vec : ", a_vec, a_vec.shape)

        mu_v = self.net_A2C.pi(s_vec)
        mu_v = self.scale_action(mu_v)

        log_prob_v = advantage * self.calc_logprob(mu_v, self.net_A2C.logstd, a_vec)
        loss_policy_v = -log_prob_v.mean()
        print("log_prob_v", loss_policy_v, loss_policy_v.shape)
        exit()
        loss = -(log_prob_v * advantage.detach()).mean() + \
               F.smooth_l1_loss(self.net_A2C.V(s_vec).reshape(-1), td_target_vec)

        print("loss ", loss)
        exit()

    def calc_logprob(self, mu_v, logstd_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
        return p1 + p2
