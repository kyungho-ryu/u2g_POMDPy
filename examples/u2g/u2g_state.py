from __future__ import print_function
from builtins import range
from pomdpy.discrete_mapping_pomdp import DiscreteState
from examples.u2g.util import getLocStat
from examples.u2g.network_config import Config
from examples.u2g.u2g_reward import U2GReward
import logging, hashlib

class U2GState(DiscreteState):
    """
    The state contains the position of the UAVs, as well as the number of GMU of each cell.

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, uav_position, gmu_position, uavs, gmus):
        self.logger  = logging.getLogger('POMDPy.State')
        self.logger.setLevel("INFO")
        self.uav_position = uav_position # list of uav cell position
        self.gmu_position = gmu_position  # list of gmu cell position
        self.uavs = uavs
        self.gmus = gmus
        self.gmuStatus = getLocStat(self.gmus, Config.MAX_GRID_INDEX)
        self.uavStatus = getLocStat(self.uavs, Config.MAX_GRID_INDEX)

        self.reward_for_actions = {}

    # def __del__(self):
    #     print(f'state destroy {id(self)}')

    def set_reward(self, key, totalDnRate, totalA2GEnergy=0, totalA2AEnergy=0, totalPropEnergy=0, NumActiveUav=0):
        self.reward_for_actions[key] = U2GReward(
            totalDnRate, totalA2GEnergy, totalA2AEnergy, totalPropEnergy, NumActiveUav
        )

    def check_already_exisit_reward(self, key):
        if key in self.reward_for_actions :
            return True
        else :
            return False

    def get_reward(self, key):
        return self.reward_for_actions[key].get_reward()

    def get_num_active_uavs(self, key):
        return self.reward_for_actions[key].get_NumActiveUav()

    def get_gmus_prediction_length(self):
        k = []
        for i in range(len(self.gmus)) :
            k.append(self.gmus[i].k)

        return k


    def copy(self):
        return self

    def __hash__(self):
        """
        Returns a decimal value representing the binary state string
        :return:
        """
        return int(self.to_string(), 2)

    def to_string(self):
        state = "UAV deployemnt : " +str(self.uav_position) + \
                "\nPrediction position of GMU : " + str(self.gmu_position)
        return state

    def print_state(self):
        state = "UAV deployemnt : " +str(self.uav_position) + \
                "\n Prediction position of GMU : " + str(self.gmu_position)
        return state

    def as_list(self):
        _list = [self.uav_position, self.gmu_position]
        return _list

    def as_DRL_state(self):
        _list = self.uav_position + self.gmu_position
        return _list

    def get_key(self):
        return hashlib.sha256(str(self.as_list()).encode()).hexdigest()

class UAV:
    def __init__(self, _id, _x, _y, _z):
        self.id = _id
        self.x = _x
        self.y = _y
        self.h = _z
        self.cell = 0 #cell index
        self.power = 'on'
        self.bGateway = False

    def get_location(self):
        return self.x, self.y, self.cell

class GMU:
    def __init__(self, _id, _x, _y, USER_DEMAND, observed, SL_params):
        self.id = _id
        self.x = _x
        self.y = _y
        self.cell = 0
        self.dnRate = 0
        self.demand = USER_DEMAND

        self.observed = observed

        self.S = []
        self.RO = []
        self.eta = 0
        self.k = 1

        if not self.observed:
            self.set_SL_params(SL_params)

    def set_SL_params(self, SL_params):
        self.S = SL_params[0]
        self.RO = SL_params[2]
        self.eta = SL_params[3]
        self.k = SL_params[4]

    def get_SL_params(self, diameterofCell):
        return self.S, self.get_cellCoordinate(diameterofCell), self.RO, self.eta, self.k

    def get_cellCoordinate(self, diameterofCell):
        x, y = int(self.x // diameterofCell), int(self.y // diameterofCell)

        if x < 0 or y < 0:
            return -1
        else:
            return x, y

    def get_location(self):
        return self.x, self.y, self.cell