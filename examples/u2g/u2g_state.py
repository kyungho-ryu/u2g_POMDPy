from __future__ import print_function
from builtins import range
from pomdpy.discrete_pomdp import DiscreteState
from examples.u2g.util import uavTopoUpdate, getLocStat
from examples.u2g.network_config import Config
from examples.u2g.channel import calA2ALinkRate, setA2GDefaultRadioResource
import networkx as nx
import logging

class U2GState(DiscreteState):
    """
    The state contains the position of the UAVs, as well as the number of GMU of each cell.

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, uav_position, gmu_position, uavs, gmus, IS_ROOT=False):
        self.logger  = logging.getLogger('POMDPy.State')
        self.uav_position = uav_position # list of uav cell position
        self.gmu_position = gmu_position  # list
        self.uavs = uavs
        self.gmus = gmus
        self.G = nx.Graph()
        self.activeUavs = []
        self.activeUavsID = []
        self.uavStatus = getLocStat(self.uavs, Config.MAX_GRID_INDEX)
        self.gmuStatus = getLocStat(self.gmus, Config.MAX_GRID_INDEX)

        self.a2aLinkStatus = {}  # {e: {'max': 0, 'load':0}} max capa and load for each edge e.g., (1, 2)

        if not IS_ROOT :
            self.set_activeUavs()
            self.test()
    def set_activeUavs(self):
        for i in range(len(self.uavs)) :
            if self.uavs[i].power == "on" :
                self.activeUavs.append(self.uavs[i])
                self.activeUavsID.append(self.uavs[i].id)

        uavTopoUpdate(self.G, self.activeUavs, self.uavStatus, Config.MAX_XGRID_N, Config.MAX_GRID_INDEX)

    def test(self):
        self.calA2ALinkCapa()
        totalThroughput = 0
        totalCommEnergy = 0

        # predict a2g traffics
        for _cell in self.gmuStatus:
            if self.gmuStatus[_cell]:
                sUav = self.findServingUAV(self.uavStatus[_cell])
                if sUav:
                    setA2GDefaultRadioResource(sUav, self.gmuStatus[_cell])
                else:
                    self.logger.debug('There is no serving UAV in cell :{}'.format(_cell))
                    self.logger.debug('Those GMU cannot receive service : {}'.format([g.id for g in self.gmuStatus[_cell]]))


    def calA2ALinkCapa(self):
        for e in list(self.G.edges()):
            self.a2aLinkStatus[e] = {'max': 0, 'load': 0}
            self.a2aLinkStatus[e]['max'] = calA2ALinkRate(self.uavs[e[0]], self.uavs[e[1]])

        self.logger.debug("A2A Link State : {}".format(self.a2aLinkStatus))

    def findServingUAV(self, _luavs):
        for u in _luavs:
            if u.power == 'on' and u.bGateway == False:
                return u
        return None

    def get_activeUavs(self):
        return self.activeUavsID

    def draw_graph(self):
        nx.draw(self.G, with_labels=True, font_weight="bold")

    def copy(self):
        return U2GState(self.uav_position, self.gmu_position)

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