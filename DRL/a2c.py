import torch
import torch.nn as nn

HID_SIZE = 64
INNER_HID_SIZE = 32
class ActorCritic(nn.Module):
    def __init__(self, state_size, act_size):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(state_size, HID_SIZE)
        self.fc2 = nn.Linear(HID_SIZE, HID_SIZE)
        self.fc_mu = nn.Linear(HID_SIZE, act_size)
        # self.fc_mu1 = nn.Linear(HID_SIZE, INNER_HID_SIZE)
        # self.fc_mu2 = nn.Linear(INNER_HID_SIZE, act_size)
        # self.fc_v1 = nn.Linear(HID_SIZE, INNER_HID_SIZE)
        # self.fc_v2 = nn.Linear(INNER_HID_SIZE, 1)
        self.fc_v = nn.Linear(HID_SIZE, 1)
        self.logstd = nn.Parameter(torch.zeros(act_size))
        self.act_size = act_size

    def pi(self, x):
        fc1 = torch.tanh(self.fc1(x))
        fc2 = torch.tanh(self.fc2(fc1))
        # fc_mu1 = torch.tanh(self.fc_mu1(fc2))
        mu = torch.tanh(self.fc_mu(fc2))

        return mu

    def V(self, x):
        fc1 = torch.tanh(self.fc1(x))
        fc2 = torch.tanh(self.fc2(fc1))
        v = (self.fc_v(fc2))
        # fc_v1 = torch.tanh(self.fc_v1(fc2))
        # v = self.fc_v2(fc_v1)

        return v
