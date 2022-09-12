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
max_train_steps = 60000
ENTROPY_BETA = 1e-3
GAMMA = 0.9
GAE_LAMBDA = 0.9
PPO_EPS = 0.2
CLIP_GRAD_NORM = -1
log_std_clip = (-1,1)
class PPOModel :
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
        # action = action + np.exp(0.4) * np.random.normal(size=logstd.shape)
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
        for i in range(sample.NumSample) :
            self.optimizer.zero_grad()
            traj_states_v = torch.FloatTensor(sample.s_list[i])
            adv_v, ref_v = self.calc_adv_ref(sample.r_list[i], traj_states_v, sample.termial_list[i])
            adv_v_list.append(np.mean(adv_v.tolist()))
            adv_v = adv_v.unsqueeze(-1)
            # print("adv_v", adv_v, adv_v.shape, adv_v.grad)
            # print("ref_v", ref_v, ref_v.shape, ref_v.grad)
            old_logprob_v = torch.FloatTensor(sample.logprob_list[i][:-1])
            # print("old_logprob_v", old_logprob_v, old_logprob_v.shape, old_logprob_v.grad)

            mu_v, std = self.net_A2C.pi(traj_states_v[:-1])
            a_vec = torch.FloatTensor(sample.a_list[i][:-1])
            a_vec = self.relex_scale(a_vec)
            logprob_pi_v = self.calc_logprob(mu_v, std, a_vec)
            logprob_pi_v = torch.clip(logprob_pi_v, log_std_clip[0], log_std_clip[1])
            # print("logprob_pi_v", logprob_pi_v, logprob_pi_v.shape, logprob_pi_v.grad)

            ratio_v = torch.exp(logprob_pi_v - old_logprob_v)
            # print("ratio_v", ratio_v, ratio_v.shape, ratio_v.grad)

            surr_obj_v = adv_v * ratio_v
            clipped_surr_v = adv_v * torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
            loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()

            # print("surr_obj_v", surr_obj_v, surr_obj_v.shape, surr_obj_v.grad)
            # print("clipped_surr_v", clipped_surr_v, clipped_surr_v.shape, clipped_surr_v.grad)
            # print("loss_policy_v", loss_policy_v, loss_policy_v.shape, loss_policy_v.grad)

            value = self.net_A2C.V(traj_states_v[:-1]).reshape(-1)
            loss = loss_policy_v + F.smooth_l1_loss(value, ref_v)
            loss_v_list.append(np.mean(loss.tolist()))

            # print("value ", value, value.shape)
            # print("loss ", loss, loss.shape)
            loss.backward()
            if CLIP_GRAD_NORM != -1:
                torch.nn.utils.clip_grad_norm(self.net_A2C.parameters(), CLIP_GRAD_NORM)
            self.optimizer.step()

            std_list.append(float(std.mean()))

            self.step +=1

        return np.mean(adv_v_list), np.mean(loss_v_list), self.step, std_list, self.logStdStep

    def calc_logprob(self, mu_v, logstd_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
        return p1 + p2

    def calc_adv_ref(self, rewards, states_v, terminals, device="cpu"):
        """
        By trajectory calculate advantage and 1-step ref value
        :param trajectory: trajectory list
        :param net_crt: critic network
        :param states_v: states tensor
        :return: tuple with advantage numpy array and reference values
        """
        values_v = self.net_A2C.V(states_v)
        values = values_v.squeeze().data.cpu().numpy()
        # generalized advantage estimator: smoothed version of the advantage
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for val, next_val, reward, terminal in zip(reversed(values[:-1]), reversed(values[1:]),
                                         reversed(rewards[:-1]), reversed(terminals[:-1])):
            if terminal:
                delta = reward - val
                last_gae = delta
            else:
                delta = reward + GAMMA * next_val - val
                last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
        ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
        return adv_v, ref_v