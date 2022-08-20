from pomdpy.pomdp import Model, StepResult
from mobility.semi_lazy import SLModel
from examples.u2g import energy
from examples.u2g.network_config import Config
from examples.u2g.u2g_action import U2GAction
from examples.u2g.u2g_state import U2GState, UAV, GMU
from examples.u2g.u2g_observation import U2GObservation
from examples.u2g.u2g_position_history import GMUData, PositionAndGMUData
from examples.u2g.util import getA2ADist, getLocStat, getNgbCellAvail
from examples.u2g.energy import calA2GCommEnergy, calA2ACommEnergy, calA2GMaxCommEnergy, calA2AMaxCommEnergy
import logging, random, copy

from pomdpy.discrete_pomdp import DiscreteActionPool, DiscreteObservationPool


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
        # self.gmus = []
        self.uavStatus = {}
        # self.gmuStatus = {}
        self.uavPosition = [0 for _ in range(Config.MAX_GRID_INDEX + 1)]    # except gcc

        # Uav object to keep the same position every epoch
        self.init_uavs = []

        self.MaxEnergyConsumtion = 0
        self.MaxDnRate = 0

        # list of map
        self.env_map = []
        # Smart gmu data
        self.all_gmu_data = []

        # Mobility model
        self.mobility_SLModel = SLModel(Config.NUM_GMU, Config.GRID_W, Config.MAX_XGRID_N)

        self.initialize()



    # initialize the maps of the grid
    def initialize(self) :
        self.setGridCenterCoord(self.GRID_CENTER)
        self.p0, self.p1 = energy.calUavHoverEnergy()

        self.generate_UAV()
        self.set_envMap()

        # self.generate_GMU()

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

    def generate_GMU(self):

        # generate gmu according to deployment of UAV
        # include both gmus with real position and gmus with predicted position
        for i in range(Config.NUM_GMU):
            id, coordinate, observed, SL_params = self.mobility_SLModel.create_gmu_for_simulation(i, self.env_map)
            self.gmus.append(GMU(id, coordinate[0], coordinate[1], Config.USER_DEMAND, observed, SL_params))

        self.numGmus = Config.NUM_GMU
        self.logger.info("Create GMU :{}".format(self.numGmus))
        # check cell location for uav, gmu
        self.updateCellInfo(self.gmus)
        for gmu in self.gmus:
            self.logger.debug("GMU'{} - loc(x,y), cell :{}".format(gmu.id, gmu.get_location()))

        # deployment status check
        self.gmuStatus = getLocStat(self.gmus, Config.MAX_GRID_INDEX)
        self.logger.info("Gmu is deployed in cells : {}".format([len(self.gmuStatus[gs]) for gs in self.gmuStatus]))


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


    def getGridCenter(self, _gIdx):
        _yGrid = _gIdx // Config.MAX_XGRID_N
        _xGrid = _gIdx % Config.MAX_XGRID_N
        return (_xGrid * Config.GRID_W) + Config.GRID_W / 2, \
               (_yGrid * Config.GRID_W) + Config.GRID_W / 2


    def getGridIndex(self, _x, _y):
        _xGrid = int(_x // Config.GRID_W)
        _yGrid = int(_y // Config.GRID_W)
        return _yGrid * Config.MAX_XGRID_N + _xGrid

    def get_reward(self, next_state):
        totalDnRate = next_state.get_gmu_dnRate()
        totalEnergyConsumtion = next_state.get_total_energy_consumption()

        self.logger.debug("GMU's dnRate : {}".format(totalDnRate))
        self.logger.debug("TotalEnergyConsumtion : {}".format(totalEnergyConsumtion))

        totalEnergyConsumtion, totalDnRate = self.norm_rewards(totalEnergyConsumtion, totalDnRate)

        self.logger.debug("total rewards : {}/{}".format(totalEnergyConsumtion, totalDnRate))

        return totalEnergyConsumtion + totalDnRate

    def draw_env(self, map, name):
        self.logger.info("====U2G "+ name +" Deployment====")
        for row in reversed(map) :
            self.logger.info("| {} |".format(row))
        self.logger.info("==========================")


    def make_next_state(self, state, action):
        # change GMU trajectory
        _is_terminal = False
        sample_states = [0 for _ in range(Config.MAX_GRID_INDEX + 1)]
        gmus = []
        for gmu in state.gmus:
            if gmu.observed:
                cellIndex, coordinate, observed, SL_params = \
                    self.mobility_SLModel.get_gmu_locIndex(gmu.id, gmu.k)
            else:
                cellIndex, coordinate, observed, SL_params = \
                    self.mobility_SLModel.get_gmu_locIndex(gmu.id, gmu.k, gmu.get_SL_params(Config.GRID_W))

            if cellIndex == -1:
                self.logger.error("have to be terminated previously")
                print("test", gmu.id, gmu.k, gmu.get_SL_params(Config.GRID_W))
                print("test", gmu.x, gmu.y)
                exit()

            sample_states[cellIndex] += 1
            gmus.append(GMU(gmu.id, coordinate[0], coordinate[1], Config.USER_DEMAND, observed, SL_params))
            if SL_params[4] +1 > self.mobility_SLModel.get_max_path() or SL_params[5]:
                _is_terminal = True

        self.updateCellInfo(gmus)

        uavs = []
        uavs_deployment = []
        exisiting_cells = {} # cell : [index of cell, index of uav, distance]
        for i in range(len(state.uavs)) :
            if state.uavs[i].bGateway :
                uavs.append(UAV(i, 0, 0, Config.HEIGHT))
                uavs[i].bGateway = True
                uavs[i].cell = self.getGridIndex(uavs[i].x, uavs[i].y)
                self.control_power_according_to_distance(state.uavs, uavs, i, exisiting_cells)
            else :
                x, y= self.GRID_CENTER[action.UAV_deployment[i]]
                uavs.append(UAV(i, x, y, Config.HEIGHT))
                uavs[i].cell = self.getGridIndex(uavs[i].x, uavs[i].y)
                self.control_power_according_to_distance(state.uavs, uavs, i, exisiting_cells)
                uavs_deployment.append(self.getGridIndex(uavs[i].x, uavs[i].y))

        self.logger.debug("Next GMU state : {}".format(sample_states))
        self.logger.debug("Next UAV state : {}".format(uavs_deployment))

        return U2GState(uavs_deployment, sample_states, uavs, gmus, _is_terminal)


    def make_observation(self, next_state):
        observation = [None for _ in range(Config.MAX_GRID_INDEX+1)]

        for i in range(len(observation)) :
            if i in next_state.uav_position :
                observation[i] = next_state.gmu_position[i]

        self.logger.debug("Observation : {}".format(observation))
        return U2GObservation(observation)

    def make_reward(self, state, next_state):
        self.logger.debug("Previous UAV deployment : {}".format(state.uav_position))
        self.logger.debug("next UAV deployment : {}".format( next_state.uav_position))
        self.logger.debug("Active UAV : {}".format(next_state.get_activeUavs()))

        # 1. check gmu deploy and reallocate uavs during UAV_RELOC_PERIOD
        totalPropEnergy = self.calcurate_reallocate_uav_energy(
            next_state.uavs, state.uav_position, next_state.uav_position
        )

        # 2. Calcurate energy consumption
        totalA2GEnergy, totalA2AEnergy = self.calcurate_energy_consumption(next_state)

        totalEnergyConsumtion = totalA2GEnergy + totalA2AEnergy + totalPropEnergy
        self.logger.debug('total energy: {}, active UAVs: {}'.format(
            totalEnergyConsumtion, len(next_state.G.nodes()))
        )
        next_state.set_energy_consumtion(totalA2GEnergy, totalA2AEnergy, totalPropEnergy)

        totalDnRate = next_state.get_gmu_dnRate()
        self.logger.debug("GMU's dnRate : {}".format(totalDnRate))

        totalEnergyConsumtion, totalDnRate = self.norm_rewards(totalEnergyConsumtion, totalDnRate)

        self.logger.debug("total rewards : {}/{}".format(totalEnergyConsumtion, totalDnRate))

        return totalEnergyConsumtion + totalDnRate


    def norm_rewards(self, _energyConsumtion, _dnRate):
        energyConsumtion = 1 - _energyConsumtion / self.MaxEnergyConsumtion
        dnRate = _dnRate / self.MaxDnRate

        return energyConsumtion * Config.WoE, dnRate * Config.WoD

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


    def calcurate_reallocate_uav_energy(self, uavs, previous_uav_status, uav_status):
        totalPropEnergy = 0
        for i in range(len(uav_status)) :
            if 'off' != uavs[i].power :
                _vel = self.calUAVFlightSpeed(previous_uav_status[i], uav_status[i])
                if _vel == 0 :
                    totalPropEnergy += (self.p0+self.p1) * Config.UAV_RELOC_PERIOD
                else :
                    totalPropEnergy += energy.calUavFowardEnergy(self.p0, self.p1, _vel) * Config.UAV_RELOC_PERIOD

        self.logger.debug("Total prop energy : {}".format(totalPropEnergy))

        return totalPropEnergy

    def calcurate_energy_consumption(self, state):
        # 3.1 a2g energy
        totalA2GEnergy = 0
        for _u in state.uavs:
            if _u.power == 'on' and _u.bGateway == False and state.gmuStatus[_u.cell]:
                totalA2GEnergy += calA2GCommEnergy(_u, state.gmuStatus[_u.cell])

        self.logger.debug('total a2g energy: {}'.format(totalA2GEnergy))

        # 3.2 a2a energy
        totalA2AEnergy = 0
        for _u in state.uavs:
            if _u.power == 'on':
                _ngbs = list(state.G.neighbors(_u.id))
                for _ngb in _ngbs:
                    e = (_u.id, _ngb) if (_u.id, _ngb) in state.a2aLinkStatus else (_ngb, _u.id)
                    totalA2AEnergy += calA2ACommEnergy(
                        _u, state.uavs[_ngb], min(state.a2aLinkStatus[e]['max'], state.a2aLinkStatus[e]['load'])
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
                next_state[exisiting_cells[next_cell][1]].power = 'off'
                next_state[index].x = _cx
                next_state[index].y = _cy

                exisiting_cells[next_cell][1] = index
                exisiting_cells[next_cell][2] = new_dist
            else :
                next_state[index].power = 'off'
                next_state[index].x =  _cx
                next_state[index].y =  _cy


    ''' ===================================================================  '''
    '''                             Sampling                                 '''
    ''' ===================================================================  '''

    def sample_an_init_state(self):
        sample_states = [0 for _ in range(Config.MAX_GRID_INDEX+1)]
        gmus = []
        for i in range(self.numGmus):
            cellIndex, coordinate, observed, SL_params = self.mobility_SLModel.get_init_gmu_locIndex(i)
            sample_states[cellIndex] +=1
            gmus.append(GMU(i, coordinate[0], coordinate[1], Config.USER_DEMAND, observed, SL_params))

        self.logger.debug("sample init state : {}".format(sample_states))
        self.logger.debug("UAV init position : {}".format(self.uavPosition))

        return U2GState(self.uavPosition, sample_states, self.uavs, gmus, False, True)

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

    def sample_random_actions(self):
        uav_deployment = []
        for i in range(self.numUavs):
            x, y = self.generate_uav_with_duplication(Config.MAX_X - 1, Config.MAX_Y - 1)
            cellIndex = self.getGridIndex(x, y)
            uav_deployment.append(cellIndex)

        return U2GAction(uav_deployment)

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

    def update(self, state, sim_data):
        """
        Update the state of the simulator with sim_data
        :param sim_data:
        :return:
        """
        self.update_uavs_for_simulation(sim_data.next_state)
        self.update_gmus_for_simulation(state.gmus)



    def update_uavs_for_simulation(self, next_state):
        self.uavs = copy.deepcopy(next_state.uavs)
        self.uavStatus = copy.deepcopy(next_state.uavStatus)
        self.uavPosition = copy.deepcopy(next_state.uav_position)
        self.set_envMap()

    def update_gmus_for_simulation(self, gmus):
        self.mobility_SLModel.reset_NumObservedGMU()
        for i in range(self.numGmus):
            self.mobility_SLModel.update_gmu_for_simulation(i, gmus[i].get_SL_params(Config.GRID_W), self.env_map)


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
        result.reward = self.make_reward(state, result.next_state)
        result.is_terminal = self.is_terminal(result.next_state)

        return result, True

    def generate_particles(self,  n_particles):
        particles = []
        for _ in range(n_particles) :
            sample_states = [0 for _ in range(Config.MAX_GRID_INDEX + 1)]
            gmus = []
            for i in range(self.numGmus):
                cellIndex, coordinate, observed, SL_params = self.mobility_SLModel.get_init_gmu_locIndex(i)
                sample_states[cellIndex] += 1
                gmus.append(GMU(i, coordinate[0], coordinate[1], Config.USER_DEMAND, observed, SL_params))

            particles.append(U2GState(self.uavPosition, sample_states, self.uavs, gmus, False, True))
        return particles


    # used for data in belief root node
    def create_root_historical_data(self, solver):
        self.create_new_gmu_data()
        return PositionAndGMUData(self, self.uavPosition.copy(), self.all_gmu_data, solver)

    def create_new_gmu_data(self):
        self.all_gmu_data = []
        self.all_gmu_data.append(GMUData())

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

    def get_simulationResult(self, state):
        totalEnergyConsumtion = state.totalA2GEnergy + state.totalA2AEnergy + state.totalPropEnergy
        scaledEnergyConsumtion, scaledDnRate = self.norm_rewards(totalEnergyConsumtion, state.TotalDnRate)

        return [state.totalA2GEnergy, state.totalA2AEnergy, state.totalPropEnergy, totalEnergyConsumtion,
                state.TotalDnRate, scaledEnergyConsumtion, scaledDnRate,
                len(state.activeUavsID), self.mobility_SLModel.get_num_observed_GMU()]

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


    def is_terminal(self, state):
        """
        Returns true iff the given state is terminal.
        :param state:
        :return:
        """
        self.logger.debug("is terminal : {}".format(state.is_terminal))
        return state.is_terminal

    def is_valid(self, state):
        """
        Returns true iff the given state is valid
        :param state:
        :return:
        """

