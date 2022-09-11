import time, math, sys
import numpy as np
from pomdpy.pomdp import Model, StepResult
from mobility.semi_lazy import SLModel
from examples.u2g import energy
from examples.u2g.network_config import Config
from examples.u2g.u2g_action import U2GAction
from examples.u2g.u2g_state import U2GState, UAV, GMU
from examples.u2g.u2g_observation import U2GObservation
# from examples.u2g.u2g_position_history import GMUData, PositionAndGMUData
from examples.u2g.util import getA2ADist, getLocStat, getNgbCellAvail, getNgbCell, uavTopoUpdate, findServingUAV, getGMUCellTraffic, path2Edges, controlSrcRate, getServingUAV
from examples.u2g.energy import calA2GCommEnergy, calA2ACommEnergy, calA2GMaxCommEnergy, calA2AMaxCommEnergy
from examples.u2g.channel import calA2ALinkRate, setA2GDefaultRadioResource
import logging, random, copy, collections
import networkx as nx


from pomdpy.discrete_mapping_pomdp import DiscreteActionPool, DiscreteObservationPool
from collections import deque


module = "U2GModel"

class U2GModel(Model) : # Model
    def __init__(self, args):
        super(U2GModel, self).__init__(args)
        # logging utility
        self.logger = logging.getLogger('POMDPy.U2GModel')
        self.logger.setLevel("INFO")
        self.GRID_CENTER = {}

        self.p0 = 0
        self.p1 = 0

        self.n_rows = int(Config.MAX_X/Config.GRID_W)
        self.n_cols = int(Config.MAX_Y/Config.GRID_W)

        self.numUavs = Config.NUM_UAV
        self.numGmus = Config.NUM_GMU
        self.uavs = []
        self.uavStatus = {}
        # self.gmuStatus = {}
        self.uavPosition = [0 for _ in range(Config.NUM_UAV)]    # except gcc

        # Uav object to keep the same position every epoch
        self.init_uavs = []

        self.MaxEnergyConsumtion = 0
        self.MaxDnRate = 0

        # list of map
        self.env_map = []
        # Smart gmu data
        self.all_gmu_data = []

        # Mobility model
        self.mobility_SLModel = SLModel(Config.NUM_GMU, Config.GRID_W, Config.MAX_XGRID_N, args["min_particle_count"])
        self.init_prior_state = self.set_an_init_prior_state()
        self.init_observation = None
        self.initialize()


    # initialize the maps of the grid
    def initialize(self) :
        self.setGridCenterCoord(self.GRID_CENTER)
        self.p0, self.p1 = energy.calUavHoverEnergy()

        self.generate_UAV()
        self.set_envMap()

        self.MaxEnergyConsumtion = self.calcurate_max_uav_energy(Config.MAX_GRID_INDEX, Config.NUM_UAV)
        self.MaxDnRate = Config.USER_DEMAND * Config.NUM_GMU

        self.logger.info("Max Power Consumption : {}".format(self.MaxEnergyConsumtion))
        self.logger.info("Max dnRate : {}".format(self.MaxDnRate))

    def generate_uav_without_duplication(self, existingOBJ, MaxX, MaxY):
        while True:
            x, y = random.randint(0, MaxX), random.randint(0, MaxY)

            CellIndex = self.getGridIndex(x, y)
            if not CellIndex in existingOBJ:
                return x, y

    def generate_uav_with_duplication(self, MaxX, MaxY):
            return random.randint(0, MaxX), random.randint(0, MaxY)

    def generate_UAV(self):
        for i in range(Config.NUM_UAV):
            x,y = self.generate_uav_with_duplication(Config.MAX_X-1, Config.MAX_Y-1)
            self.uavs.append(UAV(i, x,y, Config.HEIGHT))

            cellIndex = self.getGridIndex(x, y)
            self.set_UAV_position(i, cellIndex)

        self.logger.info("Create UAV :{}".format(self.numUavs))

        # create a gateway uav as a GCC
        gcc = UAV(self.numUavs, 0, 0, Config.HEIGHT)  # gcc ID is NUM_UAV
        gcc.bGateway = True
        self.uavs.append(gcc)

        self.logger.info("Create GCC, id : {}".format(self.uavs[self.numUavs].id))

        # check cell location for uav
        self.updateCellInfo(self.uavs)

        self.initUAVLoc(self.uavs)  # grid center locating
        self.init_uavs = copy.deepcopy(self.uavs)

        for uav in self.uavs:
            self.logger.debug("UAV'{} - loc(x,y), cell :{}".format(uav.id, uav.get_location()))

        # deployment status check
        self.uavStatus = getLocStat(self.uavs, Config.MAX_GRID_INDEX)  # uav obj. per cell
        self.logger.debug("UAV is deployed in cells : {}".format([len(self.uavStatus[uav]) for uav in self.uavStatus]))

    def initUAVLoc(self, _luav):
        for u in _luav:
            if not u.bGateway:
                u.x, u.y = self.GRID_CENTER[u.cell]


    def updateCellInfo(self, _nodes):
        for n in _nodes:
            n.cell = self.getGridIndex(n.x, n.y)


    def setGridCenterCoord(self, _center):
        for c in range(Config.MAX_GRID_INDEX + 1):
            _center[c] = self.getGridCenter(c)


    def set_UAV_position(self, id, cell):
        self.uavPosition[id] = cell

    def set_UAV_positions(self, uavs):
        for uav in uavs :
            if uav.bGateway : continue
            self.uavPosition[uav.id] = uav.cell

    def set_envMap(self):
        envMap = []
        row = []
        for cell in self.uavStatus :
            row.append(len(self.uavStatus[cell]))

            if (cell+1) % Config.MAX_XGRID_N ==0 :
                envMap.append(row)
                row = []

        self.env_map = envMap

    def set_an_init_prior_state(self):
        sample_states = [0 for _ in range(Config.MAX_GRID_INDEX+1)]
        gmus = []
        for i in range(self.numGmus):
            cellIndex, coordinate = self.mobility_SLModel.get_init_prior_gmu_locIndex(i)
            sample_states[cellIndex] +=1
            gmus.append(GMU(i, coordinate[0], coordinate[1], Config.USER_DEMAND, True, 0))

        self.updateCellInfo(gmus)
        self.logger.debug("prior init state : {}".format(sample_states))

        return U2GState(self.uavPosition, sample_states, self.uavs, gmus)

    def set_activeUavs(self, next_state, uavStatus) :
        G = nx.Graph()
        activeUavs = []
        activeUavsID = []
        for i in range(len(next_state.uavs)):
            if next_state.uavs[i].power == "on":
                activeUavs.append(next_state.uavs[i])
                activeUavsID.append(next_state.uavs[i].id)

        self.logger.debug("Active Uavs : {}".format(activeUavsID))

        uavTopoUpdate(G, activeUavs, uavStatus, Config.MAX_XGRID_N, Config.MAX_GRID_INDEX)

        return G, activeUavs, activeUavsID

    def set_servingGmusTraffics(self, G, gmuStatus, next_state, uavStatus):
        a2aLinkStatus = self.calA2ALinkCapa(G, next_state.uavs)

        # predict a2g traffics
        NUM_non_discovery = 0
        for _cell in gmuStatus:
            if gmuStatus[_cell]:
                sUav = findServingUAV(uavStatus[_cell])
                if sUav:
                    setA2GDefaultRadioResource(sUav, gmuStatus[_cell])
                else:
                    NUM_non_discovery += len(gmuStatus[_cell])
                    self.logger.debug('There is no serving UAV in cell :{}'.format(_cell))
                    self.logger.debug('Those GMU cannot receive service : {}'.format([g.id for g in gmuStatus[_cell]]))


        if self.discovery_penalty :
            probability = (1 - NUM_non_discovery / self.numGmus)
            if self.discovery_penalty_threshold > probability :
                self.logger.debug("Over discovery penalty threshold  : {}".format(probability))

                return self.penalty, None, None

        gmuFlowsPerLink = {}
        srcUavRate = collections.defaultdict(int)  # uId: UAV backhaul rate

        for e in G.edges():
            gmuFlowsPerLink[e] = []

        connection = False
        for i in G.nodes():
            if i != Config.NUM_UAV: # except to gateway
                try :
                    _path = nx.shortest_path(G, source=i, target=Config.NUM_UAV, weight='weight')
                except :
                    self.logger.debug("{} don't have any path".format(i))
                    continue
                connection = True
                aggGmuRate = getGMUCellTraffic(next_state.uavs[i].cell, gmuStatus)
                srcUavRate[i] = aggGmuRate
                pathEdges = path2Edges(list(_path))

                self.logger.debug("{}' path : {}, aggGmuRate : {}".format(i, _path, aggGmuRate))

                # accumulate weight for next routing diversification
                for e in pathEdges:
                    if e not in list(G.edges()):
                        x1, y1 = e
                        e = (y1, x1)

                    gmuFlowsPerLink[e].append(i)
                    a2aLinkStatus[e]['load'] += aggGmuRate
                    tmp = list(dict(G.get_edge_data(*e)).values())
                    if tmp:
                        G.add_edge(e[0], e[1], weight=aggGmuRate + tmp[0])
                    else:
                        G.add_edge(e[0], e[1], weight=aggGmuRate)

        self.logger.debug("gmuFlowsPerLink : {}".format(gmuFlowsPerLink))
        self.logger.debug("a2aLinkStatus : {}".format(a2aLinkStatus))

        if connection == False and self.connection_penalty :
            self.logger.debug("There is no any connection with GCC")
            return  self.penalty, None, None

        # control sending flowRate according to limited backhaul link capacity
        for e in a2aLinkStatus:
            if a2aLinkStatus[e]['max'] < a2aLinkStatus[e]['load']:
                if len(gmuFlowsPerLink[e]) >= 1:  # multiple flows
                    controlSrcRate(srcUavRate, gmuFlowsPerLink[e], a2aLinkStatus[e]['max'])

        TotalDnRate = self.re_allocate_gmu_dnRate(gmuStatus, next_state.uavs, srcUavRate)

        return TotalDnRate, a2aLinkStatus, G


    def getGridCenter(self, _gIdx):
        _yGrid = _gIdx // Config.MAX_XGRID_N
        _xGrid = _gIdx % Config.MAX_XGRID_N
        return (_xGrid * Config.GRID_W) + Config.GRID_W / 2, \
               (_yGrid * Config.GRID_W) + Config.GRID_W / 2


    def getGridIndex(self, _x, _y):
        _xGrid = int(_x // Config.GRID_W)
        _yGrid = int(_y // Config.GRID_W)
        return _yGrid * Config.MAX_XGRID_N + _xGrid

    def get_reward(self, state, action, next_state):
        key = action.get_key()
        if state.check_already_exisit_reward(key) :
            totalA2GEnergy, totalA2AEnergy, totalPropEnergy, totalDnRate = state.get_reward(key)
            totalEnergyConsumtion = totalA2GEnergy + totalA2AEnergy + totalPropEnergy

            self.logger.debug("GMU's dnRate : {}".format(totalDnRate))
            self.logger.debug("TotalEnergyConsumtion : {}".format(totalEnergyConsumtion))

            totalEnergyConsumtion, totalDnRate = self.norm_rewards(totalEnergyConsumtion, totalDnRate)

            self.logger.debug("total rewards : {}/{}".format(totalEnergyConsumtion, totalDnRate))

            TotalReward = totalEnergyConsumtion + totalDnRate
        else :
            TotalReward = self.make_reward(state, action, next_state)

        return TotalReward



    def draw_env(self, map, name):
        self.logger.info("====U2G "+ name +" Deployment====")
        for row in reversed(map) :
            self.logger.info("| {} |".format(row))
        self.logger.info("==========================")


    def make_next_state(self, state, action):
        # change GMU trajectory
        gmus = []
        sample_states = [0 for _ in range(Config.MAX_GRID_INDEX + 1)]
        self.mobility_SLModel.update_SimulationPredictionLength()
        for i in range(self.numGmus):
            cellIndex, coordinate, k = self.mobility_SLModel.get_trajectory_for_simulation(i)

            if cellIndex == -1:
                self.logger.error("have to be terminated previously")
                exit()

            sample_states[cellIndex] += 1
            gmus.append(GMU(i, coordinate[0], coordinate[1], Config.USER_DEMAND, False, k))
        self.updateCellInfo(gmus)

        uavs, uavs_deployment = self.make_next_UAVs(state, action)

        self.logger.debug("Next GMU state : {}".format(sample_states))
        self.logger.debug("Next UAV state : {}".format(uavs_deployment))
        return U2GState(uavs_deployment, sample_states, uavs, gmus)

    def make_next_real_state(self, uavs, uavs_deployment):
        # change GMU trajectory
        gmus = []
        sample_states = [0 for _ in range(Config.MAX_GRID_INDEX + 1)]
        for i in range(self.numGmus):
            cellIndex, coordinate, k = self.mobility_SLModel.get_trajectory_for_simulation(i)

            if cellIndex == -1:
                self.logger.error("have to be terminated previously")
                exit()

            sample_states[cellIndex] += 1
            gmus.append(GMU(i, coordinate[0], coordinate[1], Config.USER_DEMAND, False, k))
        self.updateCellInfo(gmus)

        self.logger.debug("Next GMU state : {}".format(sample_states))
        self.logger.debug("Next UAV state : {}".format(uavs_deployment))

        return U2GState(uavs_deployment, sample_states, uavs, gmus)

    def make_observation(self, next_state):
        observation = [None for _ in range(Config.MAX_GRID_INDEX+1)]

        for i in range(len(observation)) :
            if i in next_state.uav_position :
                observation[i] = next_state.gmu_position[i]

        self.logger.debug("Observation : {}".format(observation))
        return U2GObservation(observation)

    def make_reward(self, state, action, next_state):
        self.logger.debug("Previous UAV deployment : {}".format(state.uav_position))
        self.logger.debug("next UAV deployment : {}".format( next_state.uav_position))

        key = action.get_key()
        if state.check_already_exisit_reward(key) :
            totalA2GEnergy, totalA2AEnergy, totalPropEnergy, totalDnRate = state.get_reward(key)
            totalEnergyConsumtion = totalA2GEnergy + totalA2AEnergy + totalPropEnergy
            self.logger.debug("Already exisit reward for an action, Reward: {}/{}".format(totalEnergyConsumtion, totalDnRate))
        else :
            # 1. check gmu deploy and reallocate uavs during UAV_RELOC_PERIOD
            totalPropEnergy = self.calcurate_reallocate_uav_energy(state.uav_position, next_state.uav_position)

            gmuStatus = getLocStat(state.gmus, Config.MAX_GRID_INDEX)
            uavStatus = getLocStat(next_state.uavs, Config.MAX_GRID_INDEX)

            # 2. Calcurate energy consumption
            G, activeUavs, activeUavsID = self.set_activeUavs(next_state, uavStatus)
            totalDnRate, a2aLinkStatus, G = self.set_servingGmusTraffics(G, gmuStatus, next_state, uavStatus)
            self.logger.debug("GMU's dnRate : {}".format(totalDnRate))

            if totalDnRate == self.penalty :
                state.set_reward(key, self.penalty)

                return self.penalty

            totalA2GEnergy, totalA2AEnergy = self.calcurate_energy_consumption(
                gmuStatus, next_state, a2aLinkStatus, G
            )

            totalEnergyConsumtion = totalA2GEnergy + totalA2AEnergy + totalPropEnergy
            self.logger.debug('total energy: {}, active UAVs: {}'.format(
                totalEnergyConsumtion, len(G.nodes()))
            )
            state.set_reward(key, totalDnRate, totalA2GEnergy,
                             totalA2AEnergy, totalPropEnergy, len(activeUavsID))

        totalEnergyConsumtion, totalDnRate = self.norm_rewards(totalEnergyConsumtion, totalDnRate)

        self.logger.debug("total rewards : {}/{}".format(totalEnergyConsumtion, totalDnRate))

        return totalEnergyConsumtion + totalDnRate

    def make_next_UAVs(self, state, action):
        uavs = []
        uavs_deployment = []
        exisiting_cells = {}  # cell : [index of cell, index of uav, distance]
        for i in range(len(state.uavs)):
            if state.uavs[i].bGateway:
                uavs.append(UAV(i, 0, 0, Config.HEIGHT))
                uavs[i].bGateway = True
                uavs[i].cell = self.getGridIndex(uavs[i].x, uavs[i].y)
                self.control_power_according_to_distance(state.uavs, uavs, i, exisiting_cells)
            else:
                x, y = self.GRID_CENTER[action.UAV_deployment[i]]
                uavs.append(UAV(i, x, y, Config.HEIGHT))
                uavs[i].cell = self.getGridIndex(uavs[i].x, uavs[i].y)
                self.control_power_according_to_distance(state.uavs, uavs, i, exisiting_cells)
                uavs_deployment.append(self.getGridIndex(uavs[i].x, uavs[i].y))

        return uavs, uavs_deployment

    def make_current_real_gmuStatus(self):
        sample_states = [0 for _ in range(Config.MAX_GRID_INDEX + 1)]
        gmus = []
        for i in range(self.numGmus):
            cellIndex, coordinate = self.mobility_SLModel.get_real_locIndex(i)
            sample_states[cellIndex] += 1
            gmus.append(GMU(i, coordinate[0], coordinate[1], Config.USER_DEMAND, True, 0))

        # index, self.MOS[id].get_location(), True, None
        self.updateCellInfo(gmus)

        gmuStatus = getLocStat(gmus, Config.MAX_GRID_INDEX)

        return gmuStatus


    def make_reward_for_realState(self, state, next_state, gmuStatus):
        self.logger.debug("Previous UAV deployment : {}".format(state.uav_position))
        self.logger.debug("next UAV deployment : {}".format(next_state.uav_position))

        totalPropEnergy = self.calcurate_reallocate_uav_energy(state.uav_position, next_state.uav_position)

        uavStatus = getLocStat(next_state.uavs, Config.MAX_GRID_INDEX)

        # 2. Calcurate energy consumption
        G, activeUavs, activeUavsID = self.set_activeUavs(next_state, uavStatus)

        # gmuStatus, next_state, uavStatus
        totalDnRate, a2aLinkStatus, G = self.set_servingGmusTraffics(G, gmuStatus, next_state, uavStatus)
        self.logger.debug("GMU's dnRate : {}".format(totalDnRate))

        if totalDnRate == self.penalty:
            return self.penalty, 0

        totalA2GEnergy, totalA2AEnergy = self.calcurate_energy_consumption(
            gmuStatus, next_state, a2aLinkStatus, G
        )

        totalEnergyConsumtion = totalA2GEnergy + totalA2AEnergy + totalPropEnergy
        self.logger.debug('total energy: {}, active UAVs: {}'.format(
            totalEnergyConsumtion, len(G.nodes()))
        )

        totalEnergyConsumtion, totalDnRate = self.norm_rewards(totalEnergyConsumtion, totalDnRate)

        self.logger.debug("total rewards : {}/{}".format(totalEnergyConsumtion, totalDnRate))

        return totalEnergyConsumtion, totalDnRate

    def norm_rewards(self, _energyConsumtion, _dnRate):
        energyConsumtion =  -1 * (_energyConsumtion / self.MaxEnergyConsumtion)
        energyConsumtion =  1 - (_energyConsumtion / self.MaxEnergyConsumtion)
        dnRate = _dnRate / self.MaxDnRate

        return energyConsumtion * Config.WoE, dnRate * Config.WoD

    def re_allocate_gmu_dnRate(self, gmuStatus, uavs, srcUavRate):
        _dnRate = 0
        for _cell in gmuStatus:
            if gmuStatus[_cell]:
                _uav = getServingUAV(_cell, uavs)
                if _uav == None :
                    continue

                _rate = srcUavRate[_uav.id] / len(gmuStatus[_cell])
                for g in gmuStatus[_cell]:
                    g.dnRate = _rate
                    _dnRate += _rate

        TotalDnRate = _dnRate

        return TotalDnRate

    def calA2ALinkCapa(self, G, uavs):
        a2aLinkStatus = {}
        for e in list(G.edges()):
            a2aLinkStatus[e] = {'max': 0, 'load': 0}
            a2aLinkStatus[e]['max'] = calA2ALinkRate(uavs[e[0]], uavs[e[1]])

        self.logger.debug("A2A Link State : {}".format(a2aLinkStatus))

        return a2aLinkStatus

    def calcurate_max_uav_energy(self, MAX_GRID_INDEX, NUM_UAV):
        totalPropEnergy = 0
        opposit_index = MAX_GRID_INDEX

        for i in range(MAX_GRID_INDEX+1) :
            _vel = self.calUAVFlightSpeed(i, opposit_index)
            if _vel == 0 :
                totalPropEnergy += (self.p0 + self.p1) * Config.UAV_RELOC_PERIOD
            else:
                totalPropEnergy += energy.calUavFowardEnergy(self.p0, self.p1, _vel) * Config.UAV_RELOC_PERIOD
            opposit_index -=1

            if i == (NUM_UAV -1) : break

        totalA2GEnergy = self.numUavs * calA2GMaxCommEnergy()
        totalA2AEnergy = 0
        for i in range(self.numUavs) :
            NgbCell = getNgbCellAvail(i, Config.MAX_XGRID_N, Config.MAX_GRID_INDEX)
            for cell in NgbCell :
                if cell :
                    totalA2AEnergy += calA2AMaxCommEnergy()

        totalPowerConsumption = totalPropEnergy + totalA2GEnergy + totalA2AEnergy

        return totalPowerConsumption


    def calcurate_reallocate_uav_energy(self, previous_uav_status, uav_status):
        totalPropEnergy = 0
        for i in range(len(uav_status)) :
            # if 'off' != uavs[i].power :
            _vel = self.calUAVFlightSpeed(previous_uav_status[i], uav_status[i])
            if _vel == 0 :
                totalPropEnergy += (self.p0+self.p1) * Config.UAV_RELOC_PERIOD
            else :
                totalPropEnergy += energy.calUavFowardEnergy(self.p0, self.p1, _vel) * Config.UAV_RELOC_PERIOD

        self.logger.debug("Total prop energy : {}".format(totalPropEnergy))

        return totalPropEnergy

    def calcurate_energy_consumption(self, gmuStatus, next_state, a2aLinkStatus, G):
        # 3.1 a2g energy
        totalA2GEnergy = 0
        for _u in next_state.uavs:
            if _u.power == 'on' and _u.bGateway == False and gmuStatus[_u.cell]:
                totalA2GEnergy += calA2GCommEnergy(_u, gmuStatus[_u.cell])

        self.logger.debug('total a2g energy: {}'.format(totalA2GEnergy))

        # 3.2 a2a energy
        totalA2AEnergy = 0
        for _u in next_state.uavs:
            if _u.power == 'on':
                _ngbs = list(G.neighbors(_u.id))
                for _ngb in _ngbs:
                    e = (_u.id, _ngb) if (_u.id, _ngb) in a2aLinkStatus else (_ngb, _u.id)
                    totalA2AEnergy += calA2ACommEnergy(
                        _u, next_state.uavs[_ngb], min(a2aLinkStatus[e]['max'], a2aLinkStatus[e]['load'])
                    )

        self.logger.debug('total a2a energy: {}'.format(totalA2AEnergy))

        return totalA2GEnergy, totalA2AEnergy

    def calUAVFlightSpeed(self, _tidx, _cIdx):  # target, current
        _tx, _ty = self.GRID_CENTER[_tidx]
        _cx, _cy = self.GRID_CENTER[_cIdx]
        _d = getA2ADist(_tx, _ty, _cx, _cy)
        return _d / Config.UAV_RELOC_PERIOD

    def control_power_according_to_distance(self, current_state, next_state, index, exisiting_cells):
        current_cell = current_state[index].cell
        next_cell = next_state[index].cell
        _cx, _cy = self.GRID_CENTER[current_cell]
        _nx, _ny = self.GRID_CENTER[next_cell]

        if next_cell not in exisiting_cells.keys() :
            exisiting_cells[next_cell] = [next_cell, index, getA2ADist(_cx, _cy, _nx, _ny)]
        else :
            new_dist = getA2ADist(_cx, _cy, _nx, _ny)
            if new_dist < exisiting_cells[next_cell][2] :
                next_state[exisiting_cells[next_cell][1]].x = _cx
                next_state[exisiting_cells[next_cell][1]].y = _cy
                next_state[exisiting_cells[next_cell][1]].power = 'off'

                exisiting_cells[next_cell][1] = index
                exisiting_cells[next_cell][2] = new_dist
            else :
                next_state[index].x = _cx
                next_state[index].y = _cy
                next_state[index].power = 'off'


    ''' ===================================================================  '''
    '''                             Sampling                                 '''
    ''' ===================================================================  '''

    def sample_an_init_observation(self):
        if self.init_observation == None :
            observation = [None for _ in range(Config.MAX_GRID_INDEX + 1)]
            gmu_position = self.mobility_SLModel.get_gmu_position(Config.MAX_GRID_INDEX + 1)
            for i in range(len(observation)):
                if i in self.uavPosition:
                    observation[i] = gmu_position[i]

            self.logger.debug("Observation : {}".format(observation))
            self.init_observation = U2GObservation(observation)

        return self.init_observation


    def sample_an_init_state(self):
        sample_states = [0 for _ in range(Config.MAX_GRID_INDEX+1)]
        gmus = []
        for i in range(self.numGmus):
            cellIndex, coordinate, observed, k = self.mobility_SLModel.get_init_gmu_locIndex(i)
            sample_states[cellIndex] +=1
            gmus.append(GMU(i, coordinate[0], coordinate[1], Config.USER_DEMAND, observed, k))
        self.updateCellInfo(gmus)

        self.logger.debug("sample init state : {}".format(sample_states))
        self.logger.debug("UAV init position : {}".format(self.uavPosition))

        return U2GState(self.uavPosition, sample_states, self.uavs, gmus)

    def sample_state_uninformed(self):
        """
        Samples a state from a poorly-informed prior. This is used by the provided default
        implementation of the second generateParticles() method.
        :return:
        """

    def sample_state_informed(self, belief):
        """
        :param belief:
        :return:
        """

    def sample_random_actions(self, exsiting_actions):
        while True :
            uav_deployment = []
            for i in range(self.numUavs):
                x, y = self.generate_uav_with_duplication(Config.MAX_X - 1, Config.MAX_Y - 1)
                cellIndex = self.getGridIndex(x, y)
                uav_deployment.append(cellIndex)

            action = U2GAction(uav_deployment)
            if action.get_key() not in exsiting_actions :
                return action

    def sample_near_actions(self, uav_deployment):
        new_uav_deployment = []

        _locStat = [i for i in range(Config.MAX_GRID_INDEX+1)]
        for i in range(len(uav_deployment)) :
            candidate = getNgbCell(uav_deployment[i], _locStat, Config.MAX_XGRID_N, Config.MAX_GRID_INDEX)
            cellIndex = random.choice(candidate)

            new_uav_deployment.append(cellIndex)

        return U2GAction(new_uav_deployment)

    # def sample_gmus(self):
    #     return [len(self.gmuStatus[gs]) for gs in self.gmuStatus]

    ''' ===================================================================  '''
    '''                 Implementation of abstract Model class               '''
    ''' ===================================================================  '''

    def reset_for_simulation(self):
        """
        The Simulator (Model) should be reset before each simulation
        :return:

        """
        self.mobility_SLModel.set_simulation_state()
        # for gmu in gmus :
        #     if gmu.observed :
        #         self.mobility_SLModel.set_simulation_state(gmu.id, None)
        #     else :
        #         self.mobility_SLModel.set_simulation_state(gmu.id, gmu.get_SL_params(Config.GRID_W))


    def reset_for_epoch(self):
        # reset UAV status to same init position
        self.uavs = copy.deepcopy(self.init_uavs)
        self.uavStatus = getLocStat(self.uavs, Config.MAX_GRID_INDEX)  # uav obj. per cell
        self.set_UAV_positions(self.uavs)
        self.set_envMap()

        self.draw_env(self.env_map, "UAV")

        # reset GMU status
        self.mobility_SLModel.reset(self.env_map)
        gmu_map = self.mobility_SLModel.get_gmu_env(Config.MAX_XGRID_N, Config.MAX_YGRID_N)
        self.draw_env(gmu_map, "GMU")

    def update(self, sim_data):
        """
        Update the state of the simulator with sim_data
        :param sim_data:
        :return:
        """
        # self.update_uavStatus(sim_data.next_state)
        # self.update_gmuStatus()
        pass


    def update_uavStatus(self, uavs, uav_position):
        self.uavs = copy.deepcopy(uavs)
        self.uavStatus = getLocStat(uavs, Config.MAX_GRID_INDEX)
        self.uavPosition = copy.deepcopy(uav_position)
        self.set_envMap()

    def update_gmuStatus(self):
        self.mobility_SLModel.reset_NumObservedGMU()
        for i in range(self.numGmus):
            self.mobility_SLModel.update_gmuStatus(i, self.env_map)

    def update_gmu_for_simulation(self):
        self.mobility_SLModel.update_SimulationPredictionLength()
        for i in range(self.numGmus):
            self.mobility_SLModel.update_trajectory_for_simulation(i)

    def generate_step(self, state, action):
        """
        Generates a full StepResult, including the next state, an observation, and the reward
        *
        * For convenience, the action taken is also included in the result, as well as a flag for
        * whether or not the resulting next state is terminal.
        :param state:
        :param action:
        :return: StepResult
        """

        if action is None:
            self.logger.error("Tried to generate a step with a null action")
            return None

        result = StepResult()
        result.next_state = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(result.next_state)
        result.reward = self.make_reward(state, action, result.next_state)
        result.is_terminal = self.is_terminal()

        return result, True

    def generate_real_step(self, state, action):
        """
        Generates a full StepResult, including the next state, an observation, and the reward
        *
        * For convenience, the action taken is also included in the result, as well as a flag for
        * whether or not the resulting next state is terminal.
        :param state:
        :param action:
        :return: StepResult
        """

        if action is None:
            self.logger.error("Tried to generate a step with a null action")
            return None
        uavs, uavs_deployment = self.make_next_UAVs(state, action)
        gmuStatus = self.make_current_real_gmuStatus()
        self.update_uavStatus(uavs, uavs_deployment)
        self.update_gmuStatus()

        result = StepResult()
        result.next_state = self.make_next_real_state(uavs, uavs_deployment)
        result.action = action.copy()
        result.observation = self.make_observation(result.next_state)
        result.reward = self.make_reward(state, action, result.next_state)
        totalEnergyConsumtion, totalDnRate= self.make_reward_for_realState(state, result.next_state, gmuStatus)
        # result.reward = totalEnergyConsumtion + totalDnRate
        result.is_terminal = self.is_terminal()

        return result, True, totalEnergyConsumtion, totalDnRate


    def generate_particles(self):
        sample_states = [0 for _ in range(Config.MAX_GRID_INDEX + 1)]
        gmus = []
        for i in range(self.numGmus):
            cellIndex, coordinate, observed, k = self.mobility_SLModel.get_init_gmu_locIndex(i)
            sample_states[cellIndex] += 1
            gmus.append(GMU(i, coordinate[0], coordinate[1], Config.USER_DEMAND, observed, k))
        self.updateCellInfo(gmus)

        return U2GState(self.uavPosition, sample_states, self.uavs, gmus)


    # used for data in belief root node
    def create_root_historical_data(self, solver):
        pass
        # self.create_new_gmu_data()
        # return PositionAndGMUData(self, self.uavPosition.copy(), self.all_gmu_data, solver)

    def create_new_gmu_data(self):
        pass
        # self.all_gmu_data = []
        # self.all_gmu_data.append(GMUData())

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def belief_update(self, old_belief, action, observation):
        """
        Use bayes filter to update belief distribution
        :param old_belief:
        :param action
        :param observation
        :return:
        """

    def test22(self):
        for i in range(self.numGmus) :
            real = self.mobility_SLModel.get_real_trajectory(i)
            print(real)
        exit()

    def get_dissimilarity_of_gmu_prediction(self, gmus):
        error = []
        for gmu in gmus :
            prediction_cell = gmu.get_cellCoordinate(Config.GRID_W)
            real_cell = self.mobility_SLModel.get_real_trajectory(gmu.id)
            error_x = prediction_cell[0] - real_cell[0]
            error_y = prediction_cell[1] - real_cell[1]

            error.append(math.sqrt(math.pow(error_x, 2) + math.pow(error_y, 2)))

        if error == [] :
            return 0
        else :
            return np.mean(error)


    def get_simulationResult(self, state, action):
        key = action.get_key()
        totalA2GEnergy, totalA2AEnergy, totalPropEnergy, totalDnRate = state.get_reward(key)
        totalEnergyConsumtion = totalA2GEnergy + totalA2AEnergy + totalPropEnergy
        scaledEnergyConsumtion, scaledDnRate = self.norm_rewards(totalEnergyConsumtion, totalDnRate)

        NumActiveUav = state.get_num_active_uavs(key)
        avgDnRage = totalDnRate / Config.NUM_GMU / pow(10, 3)
        return [totalA2GEnergy, totalA2AEnergy, totalPropEnergy, totalEnergyConsumtion,
                avgDnRage, scaledEnergyConsumtion, scaledDnRate, NumActiveUav,
                self.mobility_SLModel.get_num_observed_GMU()]

    def get_initial_belief_state(self):
        """
        Return an np.array of initial belief probabilities for each state
        :return:
        """
        pass

    def get_all_states(self):
        """
        :return: list of enumerated states (discrete) or range of states (continuous)
        """

    def get_state_space(self):
        return [0, Config.MAX_GRID_INDEX]

    def get_state_dimension(self):
        return (Config.NUM_UAV)+ (Config.MAX_GRID_INDEX + 1)

    def get_all_actions(self):
        """
        :return: initial action list which is updated when child belief node is created
        """
        return []

    def get_actionClass(self):
        """
        :return: initial action list which is updated when child belief node is created
        """
        return U2GAction

    def get_action_space(self):
        return [0, Config.MAX_GRID_INDEX]

    def get_action_dimension(self):
        return Config.NUM_UAV

    def get_all_observations(self):
        """
        :return: list of enumerated observations (discrete) or range of observations (continuous)
        """

    def get_legal_actions(self, state):
        """
        Given the current state of the system, return all legal actions
        :return: list of legal actions
        """

    def get_max_undiscounted_return(self):
        """
        Calculate and return the highest possible undiscounted return
        :return:
        """


    def get_an_init_prior_state(self):
        return self.init_prior_state

    def is_terminal(self):
        """
        Returns true iff the given state is terminal.
        :param state:
        :return:
        """
        is_terminal = self.mobility_SLModel.simulation_terminal
        self.logger.debug("is terminal : {}".format(is_terminal))
        return is_terminal

    def is_valid(self, state):
        """
        Returns true iff the given state is valid
        :param state:
        :return:
        """
