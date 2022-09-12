import numpy as np
from gym import spaces
from DRL.a2c import ActorCritic
from DRL.structure import DRLType
from examples.u2g.u2g_action import U2GAction

import torch.optim as optim
import torch, logging, math
import torch.nn.functional as F

# Hyperparameters
learning_rate = 0.0002
gamma = 0.9
max_train_steps = 60000
ENTROPY_BETA = 1e-3
log_std_clip = (-1,1)
class DRLModel :
    def __init__(self, state_dim, state_space, action_dim, action_space, _DRLType):
        self.logger = logging.getLogger('POMDPy.DRLModel')
        self.logger.setLevel("INFO")
        self.state_dim = state_dim
        self.state_high = np.array([np.float32(state_space[1])] * self.state_dim)
        self.state_low = np.array([np.float32(state_space[0])] * self.state_dim)
        self.state_space = spaces.Box(self.state_low, self.state_high, dtype=np.float32)


        self.action_dim = action_dim
        self.action_high = np.array([np.float32(action_space[1])] * self.action_dim)
        self.action_low = np.array([np.float32(action_space[0])] * self.action_dim)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        self.logger.info("State dimension, space : {}, {}".format((self.state_low[0], self.state_high[0]), self.state_space.shape[0]))
        self.logger.info("action dimension, space : {}, {}".format((self.action_low[0], self.action_high[0]), self.action_space.shape[0]))

        self.net_A2C = ActorCritic(self.state_space.shape[0], self.action_space.shape[0])
        self.optimizer = optim.Adam(self.net_A2C.parameters(), lr=learning_rate)

        self.step = 0
        self.logStdStep = 0

    def get_action(self, state):
        # state = np.array(np.arange(0, 25, 0.5))
        state = torch.from_numpy(state).float()
        mu_v, std = self.net_A2C.pi(state)
        mu = mu_v.data.cpu().numpy()
        mu = self.scale_action(mu)
        logstd = std.detach().numpy()
        logstd = np.clip(logstd, log_std_clip[0], log_std_clip[1])

        action = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        action = np.clip(action, self.action_low[0], self.action_high[0])
        action = np.round(action).astype(int)

        a_vec = self.relex_scale(action)
        logprob_pi_v = self.calc_logprob(mu_v, std, torch.from_numpy(a_vec).float())
        logprob_pi_v = logprob_pi_v.detach().numpy()

        return U2GAction(list(action)), logprob_pi_v


    def scale_action(self, x) :
        # return (2/(1+np.exp(-2*x))) -1
        # return np.exp(3*x)
        return (self.action_high[0] * x + self.action_high[0]) / 2


    def relex_scale(self, x):
        return (x*2 - self.action_high[0]) / self.action_high[0]

    def get_value(self, state):
        state = torch.from_numpy(state).float()
        value = self.net_A2C.V(state)
        value = value.data.cpu().numpy()[0]

        return value

    def update(self, sample):
        adv_v_list = []
        loss_v_list = []
        std_list = []
        # print("sample.NumSample", sample.NumSample)
        for i in range(sample.NumSample):
            self.optimizer.zero_grad()
            traj_states_v = torch.FloatTensor(sample.s_list[i])
            traj_action_v = torch.FloatTensor(sample.a_list[i])

            td_target = self.compute_target(traj_states_v[-1], sample.r_list[i], sample.termial_list[i])
            # print("td_target", td_target)

            value = self.net_A2C.V(traj_states_v).reshape(-1)
            # print("value", value)
            advantage = (td_target - value).unsqueeze(dim=-1)
            adv_v_list.append(float(advantage.mean()))
            # print("advantage", advantage)

            mu_v, std = self.net_A2C.pi(traj_states_v)
            a_vec = self.relex_scale(traj_action_v)
            logprob_pi_v = self.calc_logprob(mu_v, std, a_vec)
            logprob_pi_v = torch.clip(logprob_pi_v, log_std_clip[0], log_std_clip[1])

            log_prob_v = advantage.detach() * logprob_pi_v
            # print("log_prob_v", log_prob_v)
            loss_policy_v = -1 * log_prob_v.mean()
            # print("loss_policy_v", loss_policy_v)

            loss = loss_policy_v + F.smooth_l1_loss(value, td_target)
            loss_v_list.append(float(loss.mean()))
            loss.backward()

            self.optimizer.step()

            std_list.append(float(std.mean()))

            self.step +=1

        return np.mean(adv_v_list), np.mean(loss_v_list), self.step, std_list, self.logStdStep

    def calc_logprob(self, mu_v, logstd_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
        return p1 + p2

    def compute_target(self, v_final, r_lst, mask_lst):
        G = self.net_A2C.V(v_final).detach().tolist()[0]
        td_target = list()
        for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
            if mask :
                mask = 1
            else :
                mask = 0

            G = r + gamma * G * mask
            td_target.append(G)

        return torch.FloatTensor(td_target[::-1])