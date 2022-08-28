from collections import deque
from .mobility_config import MConfig


# Moving Object
class MO() :
    def __init__(self, id):
        self.id = id
        self.x = 0
        self.y = 0
        self.cell_index_of_x = -1
        self.cell_index_of_y = -1
        self.current_t = 0
        self.backward_traj = deque(maxlen=MConfig.BH)
        self.candidate_backward_traj = deque(maxlen=5)

        self.observed = True
        self.k =  0
        # self.tNLoc = []
        # self.S = []
        # self.RO = []
        # self.eta = 0


    def update_location(self, real_location, cell_location):
        self.x = real_location[0]
        self.y = real_location[1]
        self.cell_index_of_x = cell_location[0]
        self.cell_index_of_y = cell_location[1]
        self.current_t += 1
        self.backward_traj.append(cell_location)
        self.candidate_backward_traj.append(cell_location)

    def get_location(self):
        return [self.x, self.y]

    def get_cell_location(self):
        if self.cell_index_of_x == -1 and self.cell_index_of_y == -1 :
            return []
        else :
            return self.cell_index_of_x, self.cell_index_of_y

    def get_current_time(self):
        return self.current_t

    def set_observed(self, new_status):
        self.observed = new_status

    def update_prediction_length(self):
        self.k +=1

    def reset_prediction(self):
        self.observed = True
        self.k = 0

    # def set_prediction(self, _S, _tNLoc, _RO, _eta, _k):
    #     self.S = _S
    #     self.tNLoc = _tNLoc
    #     self.RO = _RO
    #     self.eta = _eta
    #     self.k = _k
    #     self.observed = False
    #
    # def get_mobility_model(self):
    #     if self.S == [] :
    #         return None
    #     else :
    #         return self.S, self.tNLoc, self.RO, self.eta, self.k
    #


# class GMU(MO) :
#     def __init__(self, _id):
#         super(GMU, self).__init__(_id)
#         self.x = 0
#         self.y = 0
#         self.cell = 0
#         self.dnRate = 0
#
#         self.observed = True
#         self.S = []
#         self.RO = []
#         self.eta = 0
#         self.k = 0
#
#
#     def get_location(self):
#         return self.x, self.y
#
#     def set_prediction(self, _S, _RO, _eta, _k):
#         self.S = _S
#         self.RO = _RO
#         self.eta = _eta
#         self.k = _k
#         self.observed = False
#

#
#     def get_mobility_model(self):
#         if self.observed :
#             return None
#         else :
#             return self.S, self.RO, self.eta, self.k, self.current_loc

class State() :
    def __init__(self, states={}):
        self.states = states

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