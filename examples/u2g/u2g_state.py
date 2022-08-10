from __future__ import print_function
from builtins import range
from pomdpy.discrete_pomdp import DiscreteState

class U2GState(DiscreteState):
    """
    The state contains the position of the UAVs, as well as the number of GMU of each cell.

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, uav_position, gmu_status, uavs, gmus):
        self.uav_position = uav_position # list of uav cell position
        self.gmu_status = gmu_status  # list
        self.uavs = uavs
        self.gmus = gmus

    def copy(self):
        return U2GState(self.uav_position, self.gmu_status)

    def __hash__(self):
        """
        Returns a decimal value representing the binary state string
        :return:
        """
        return int(self.to_string(), 2)

    def to_string(self):
        state = "UAV deployemnt : " +str(self.uav_position) + \
                "\nPrediction position of GMU : " + str(self.gmu_status)
        return state

    def print_state(self):
        state = "UAV deployemnt : " +str(self.uav_position) + \
                "\n Prediction position of GMU : " + str(self.gmu_status)
        return state

    def as_list(self):
        _list = [self.uav_position, self.gmu_status]
        return _list


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

        if not self.observed :
            self.set_SL_params(SL_params)

    def set_SL_params(self, SL_params):
        self.S = SL_params[0]
        self.RO = SL_params[2]
        self.eta = SL_params[3]
        self.k = SL_params[4]

    def get_SL_params(self, diameterofCell):
        return self.get_cellCoordinate(diameterofCell), self.S, self.RO, self.eta, self.k

    def get_cellCoordinate(self, diameterofCell):
        x, y = int(self.x // diameterofCell), int(self.y // diameterofCell)

        if x < 0 or y < 0:
            return -1
        else:
            return x, y

    def get_location(self):
        return self.x, self.y, self.cell