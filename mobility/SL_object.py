from collections import deque
from .mobility_config import MConfig


# Moving Object
class MO() :
    def __init__(self, id):
        self.id = id
        self.current_loc = []
        self.current_t = 0
        self.backward_traj = deque(maxlen=MConfig.BH)

    def update_location(self, coordinate):
        self.current_loc = coordinate
        self.current_t +=1
        self.backward_traj.append(coordinate)

    def get_current_loc(self):
        return self.current_loc

class GMU(MO) :
    def __init__(self, _id, _demand):
        super(GMU, self).__init__(_id)
        self.x = 0
        self.y = 0
        self.cell = 0
        self.dnRate = 0
        self.demand = _demand

        self.observed = True
        self.S = []
        self.RO = []
        self.eta = 0
        self.k = 0

    def update_location(self, real_location, cell_location):
        self.x = real_location[0]
        self.y = real_location[1]
        self.current_loc = cell_location
        self.current_t +=1
        self.backward_traj.append(cell_location)

    def get_location(self):
        return self.x, self.y

    def set_prediction(self, _S, _RO, _eta, _k):
        self.S = _S
        self.RO = _RO
        self.eta = _eta
        self.k = _k
        self.observed = False

    def reset_prediction(self):
        self.observed = True
        self.S = []
        self.RO = []
        self.eta = 0
        self.k = 0

    def get_mobility_model(self):
        if self.observed :
            return None
        else :
            return self.S, self.RO, self.eta, self.k, self.current_loc

class State() :
    def __init__(self):
        self.states = {}

    def update(self, k, loc, cellIndex, traHashKey):
        createdNewState = False
        if k not in self.states :
            self.states[k] = {}
        if loc not in self.states[k] :
            self.states[k][loc] = _state(cellIndex)
            createdNewState = True

        self.states[k][loc].update_state(traHashKey)

        return list(self.states[k].keys()), createdNewState

    def get_key_value_of_k(self, k):
        return self.states[k].items()

class _state() :
    def __init__(self,cellIndex):
        self.cellIndex = cellIndex
        self.currentRO = []
        self.NumOfCell = 0
        self.probability = 0

    def update_state(self, traHashKey):
        self.currentRO.append(traHashKey)
        self.NumOfCell +=1

class Trajectory() :
    def __init__(self, traj):
        self.traj = traj
        self.updated_time = 0

    def get_trajectory(self, i):
        return self.traj["x"][i], self.traj["y"][i]

    def update_time(self, t):
        self.updated_time = t