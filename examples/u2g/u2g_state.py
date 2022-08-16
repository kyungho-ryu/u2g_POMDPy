from __future__ import print_function
from builtins import range
from pomdpy.discrete_pomdp import DiscreteState
from examples.u2g.util import uavTopoUpdate, getLocStat, getGMUCellTraffic, path2Edges, controlSrcRate, getServingUAV, findServingUAV
from examples.u2g.network_config import Config
from examples.u2g.channel import calA2ALinkRate, setA2GDefaultRadioResource
import networkx as nx
import logging, collections

class U2GState(DiscreteState):
    """
    The state contains the position of the UAVs, as well as the number of GMU of each cell.

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, uav_position, gmu_position, uavs, gmus, _is_terminal, IS_ROOT=False):
        self.logger  = logging.getLogger('POMDPy.State')
        self.logger.setLevel("INFO")
        self.uav_position = uav_position # list of uav cell position
        self.gmu_position = gmu_position  # list of gmu cell position
        self.uavs = uavs
        self.gmus = gmus
        self.G = nx.Graph()
        self.activeUavs = []
        self.activeUavsID = []
        self.uavStatus = getLocStat(self.uavs, Config.MAX_GRID_INDEX)
        self.gmuStatus = getLocStat(self.gmus, Config.MAX_GRID_INDEX)
        self.a2aLinkStatus = {}  # {e: {'max': 0, 'load':0}} max capa and load for each edge e.g., (1, 2)

        # set end-to-end flows in multi-hop uav network
        self.gmuFlowsPerLink = {}
        self.srcUavRate = collections.defaultdict(int)  # uId: UAV backhaul rate

        self.totalA2GEnergy = 0
        self.totalA2AEnergy = 0
        self.totalPropEnergy = 0
        self.TotalDnRate = 0

        self.is_terminal = _is_terminal

        if not IS_ROOT :
            self.set_activeUavs()
            self.set_servingGmusTraffics()

    def set_activeUavs(self):
        for i in range(len(self.uavs)) :
            if self.uavs[i].power == "on" :
                self.activeUavs.append(self.uavs[i])
                self.activeUavsID.append(self.uavs[i].id)

        self.logger.debug("Active Uavs : {}".format(self.activeUavsID))
        uavTopoUpdate(self.G, self.activeUavs, self.uavStatus, Config.MAX_XGRID_N, Config.MAX_GRID_INDEX)

    # aggregate gmu wireless traffic > routing decision (weighted shortest path) > control sending rate of each UAV based on backhaul congestion
    # > re-calculate UAV A2G capa > re-assign data rate for each GMUs > calculate comm. energy and throughput
    def set_servingGmusTraffics(self):
        self.calA2ALinkCapa()

        # predict a2g traffics
        for _cell in self.gmuStatus:
            if self.gmuStatus[_cell]:
                sUav = findServingUAV(self.uavStatus[_cell])
                if sUav:
                    setA2GDefaultRadioResource(sUav, self.gmuStatus[_cell])
                else:
                    self.logger.debug('There is no serving UAV in cell :{}'.format(_cell))
                    self.logger.debug('Those GMU cannot receive service : {}'.format([g.id for g in self.gmuStatus[_cell]]))

        self.gmuFlowsPerLink = {}
        self.srcUavRate = collections.defaultdict(int)  # uId: UAV backhaul rate

        for e in self.G.edges():
            self.gmuFlowsPerLink[e] = []

        for i in self.G.nodes():
            if i != Config.NUM_UAV: # except to gateway
                try :
                    _path = nx.shortest_path(self.G, source=i, target=Config.NUM_UAV, weight='weight')
                except :
                    self.logger.debug("{} don't have any path".format(i))
                    continue
                aggGmuRate = getGMUCellTraffic(self.uavs[i].cell, self.gmuStatus)
                self.srcUavRate[i] = aggGmuRate
                pathEdges = path2Edges(list(_path))

                self.logger.debug("{}' path : {}, aggGmuRate : {}".format(i, _path, aggGmuRate))

                # accumulate weight for next routing diversification
                for e in pathEdges:
                    if e not in list(self.G.edges()):
                        x1, y1 = e
                        e = (y1, x1)

                    self.gmuFlowsPerLink[e].append(i)
                    self.a2aLinkStatus[e]['load'] += aggGmuRate
                    tmp = list(dict(self.G.get_edge_data(*e)).values())
                    if tmp:
                        self.G.add_edge(e[0], e[1], weight=aggGmuRate + tmp[0])
                    else:
                        self.G.add_edge(e[0], e[1], weight=aggGmuRate)

        self.logger.debug("gmuFlowsPerLink : {}".format(self.gmuFlowsPerLink))
        self.logger.debug("a2aLinkStatus : {}".format(self.a2aLinkStatus))

        # control sending flowRate according to limited backhaul link capacity
        for e in self.a2aLinkStatus:
            if self.a2aLinkStatus[e]['max'] < self.a2aLinkStatus[e]['load']:
                if len(self.gmuFlowsPerLink[e]) >= 1:  # multiple flows
                    controlSrcRate(self.srcUavRate, self.gmuFlowsPerLink[e], self.a2aLinkStatus[e]['max'])

        self.re_allocate_gmu_dnRate()

        self.logger.debug('total throughput : {}, average : {}'.format(
            sum(self.srcUavRate.values()), sum(self.srcUavRate.values()) / len(self.gmus))
        )

    # reallocate individual gmu dnRate based on the srcUavRate and calculate energy consumption
    def re_allocate_gmu_dnRate(self):
        _dnRate = 0
        for _cell in self.gmuStatus:
            if self.gmuStatus[_cell]:
                _uav = getServingUAV(_cell, self.uavs)
                if _uav == None :
                    continue

                _rate = self.srcUavRate[_uav.id] / len(self.gmuStatus[_cell])
                for g in self.gmuStatus[_cell]:
                    g.dnRate = _rate
                    _dnRate += _rate

        self.TotalDnRate = _dnRate

    def set_energy_consumtion(self, totalA2GEnergy, totalA2AEnergy, totalPropEnergy):
        self.totalA2GEnergy = totalA2GEnergy
        self.totalA2AEnergy = totalA2AEnergy
        self.totalPropEnergy = totalPropEnergy

    def get_gmu_dnRate(self):
        return self.TotalDnRate

    def calA2ALinkCapa(self):
        for e in list(self.G.edges()):
            self.a2aLinkStatus[e] = {'max': 0, 'load': 0}
            self.a2aLinkStatus[e]['max'] = calA2ALinkRate(self.uavs[e[0]], self.uavs[e[1]])
        self.logger.debug("A2A Link State : {}".format(self.a2aLinkStatus))


    def get_activeUavs(self):
        return self.activeUavsID

    def draw_graph(self):
        nx.draw(self.G, with_labels=True, font_weight="bold")

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