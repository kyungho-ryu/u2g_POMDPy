from pomdpy.pomdp import Model, StepResult
from mobility.semi_lazy import SLModel
from examples.u2g import energy
from examples.u2g.network_config import Config
from examples.u2g.u2g_action import U2GAction
from examples.u2g.u2g_state import U2GState, UAV, GMU
from examples.u2g.u2g_observation import U2GObservation
from examples.u2g.u2g_position_history import GMUData, PositionAndGMUData
from examples.u2g.util import getA2ADist, getLocStat
import logging, random

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

        self.numUavs = 0
        self.numGmus = 0
        self.uavs = []
        self.gmus = []
        self.uavStatus = {}
        self.gmuStatus = {}

        self.uavPosition = [0 for _ in range(Config.MAX_GRID_INDEX + 1)]    # except gcc

        # list of map
        self.env_map = []

        # Smart gmu data
        self.all_gmu_data = []

        # Actual gmu states
        # generate gmu according to deployment of UAV
        # include both gmus with real position and gmus with predicted position
        self.actual_gmu_states = []

        # Mobility model
        self.mobility_SLModel = SLModel(Config.NUM_GMU, Config.GRID_W, Config.USER_DEMAND, Config.MAX_XGRID_N)

        self.initialize()

    # initialize the maps of the grid
    def initialize(self) :
        self.setGridCenterCoord(self.GRID_CENTER)
        self.p0, self.p1 = energy.calUavHoverEnergy()

        self.generate_UAV()
        self.set_envMap()

        self.generate_GMU()

        # self.reset_for_epoch()
        # self.sample_an_init_state()
        # episode starts from time t
        # 1. check gmu deploy and reallocate uavs during UAV_RELOC_PERIOD
        totalPropEnergy = 0

    def generate_uav_without_duplication(self, existingOBJ, MaxX, MaxY):
        while True:
            x, y = random.randint(0, MaxX), random.randint(0, MaxY)

            CellIndex = self.getGridIndex(x, y)
            if not CellIndex in existingOBJ:
                return x, y

    def generate_uav_with_duplication(self, MaxX, MaxY):
            return random.randint(0, MaxX), random.randint(0, MaxY)

    def generate_UAV(self):
        uav_deployment = []
        for i in range(Config.NUM_UAV):
            x,y = self.generate_uav_with_duplication(Config.MAX_X-1, Config.MAX_Y-1)
            self.uavs.append(UAV(self.numUavs, x,y, Config.HEIGHT))

            cellIndex = self.getGridIndex(x, y)
            self.set_UAV_position(self.numUavs, cellIndex)

            self.numUavs += 1
            uav_deployment.append(cellIndex)

        self.logger.info("Create UAV :{}".format(self.numUavs))

        # create a gateway uav as a GCC
        gcc = UAV(self.numUavs, 0, 0, Config.HEIGHT)  # gcc ID is NUM_UAV
        gcc.bGateway = True
        self.uavs.append(gcc)

        self.logger.info("Create GCC, id : {}".format(self.uavs[self.numUavs].id))

        # check cell location for uav
        self.updateCellInfo(self.uavs)

        self.initUAVLoc(self.uavs)  # grid center locating

        for uav in self.uavs:
            self.logger.debug("UAV'{} - loc(x,y), cell :{}".format(uav.id, uav.get_location()))

        # deployment status check
        self.uavStatus = getLocStat(self.uavs, Config.MAX_GRID_INDEX)  # uav obj. per cell
        self.logger.debug("UAV is deployed in cells : {}".format([len(self.uavStatus[uav]) for uav in self.uavStatus]))

    def generate_GMU(self):
        # generate gmu according to deployment of UAV
        # include both gmus with real position and gmus with predicted position
        for i in range(Config.NUM_GMU):
            id, coordinate, observed, SL_params = self.mobility_SLModel.update_gmu_location(self.numGmus, self.env_map)
            self.gmus.append(GMU(id, coordinate[0], coordinate[1], Config.USER_DEMAND, observed, SL_params))

            self.numGmus += 1

        self.logger.debug("Create GMU :{}".format(self.numGmus))
        # check cell location for uav, gmu
        self.updateCellInfo(self.gmus)

        for gmu in self.gmus:
            self.logger.debug("GMU'{} - loc(x,y), cell :{}".format(gmu.id, gmu.get_location()))

        # deployment status check
        self.gmuStatus = getLocStat(self.gmus, Config.MAX_GRID_INDEX)
        self.logger.debug("Gmu is deployed in cells : {}".format([len(self.gmuStatus[gs]) for gs in self.gmuStatus]))


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


    def draw_env(self):
        self.logger.info("====U2G Network====")
        for row in reversed(self.env_map) :
            self.logger.info("| {} |".format(row))
        self.logger.info("===================")


    def make_next_state(self, state, action):
        # change GMU trajectory
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

            sample_states[cellIndex] += 1
            gmus.append(GMU(gmu.id, coordinate[0], coordinate[1], Config.USER_DEMAND, observed, SL_params))

        uavs = []
        exisiting_cells = {} # cell : [index of cell, index of uav, distance]
        for i in range(len(state.uavs)) :
            if state.uavs[i].bGateway :
                uavs.append(UAV(i, 0, 0, Config.HEIGHT))
            else :
                x, y= self.GRID_CENTER[action.UAV_deployment[i]]
                uavs.append(UAV(i, x, y, Config.HEIGHT))

            uavs[i].cell = self.getGridIndex(uavs[i].x, uavs[i].y)
            self.control_power_according_to_distance(state.uavs, uavs, i, exisiting_cells)

        self.logger.info("Next GMU state : {}".format(sample_states))
        self.logger.info("Next UAV state : {}".format(action.UAV_deployment))

        return U2GState(action.UAV_deployment, sample_states, uavs, gmus)


    def make_observation(self, action, next_state):
        observation = [None for _ in range(Config.MAX_GRID_INDEX+1)]

        for i in range(len(observation)) :
            if i in action.UAV_deployment :
                observation[i] = next_state.gmu_position[i]

        self.logger.info("Observation : {}".format(observation))
        return U2GObservation(observation)

    def make_reward(self, state, action, next_state):
        self.logger.info("Previous UAV deployment : {}".format(state.uav_position))
        self.logger.info("next UAV deployment : {}".format( action.UAV_deployment))
        self.logger.info("Active UAV : {}".format(next_state.get_activeUavs()))

        # 1. check gmu deploy and reallocate uavs during UAV_RELOC_PERIOD
        totalPropEnergy = self.calcurate_reallocate_uav_energy(next_state.uavs, state.uav_position, action.UAV_deployment)

        # 2. serving gmus traffics
        # aggregate gmu wireless traffic > routing decision (weighted shortest path)
        # > control sending rate of each UAV based on backhaul congestion
        # > re-calculate UAV A2G capa > re-assign data rate for each GMUs
        # > calculate comm. energy and throughput

        exit()


    def calcurate_reallocate_uav_energy(self, uavs, previous_uav_status, uav_status):
        totalPropEnergy = 0
        for i in range(len(uav_status)) :
            if 'off' != uavs[i].power :
                _vel = self.calUAVFlightSpeed(previous_uav_status[i], uav_status[i])
                if _vel == 0 :
                    totalPropEnergy += (self.p0+self.p1) * Config.UAV_RELOC_PERIOD
                else :
                    totalPropEnergy += energy.calUavFowardEnergy(self.p0, self.p1, _vel) * Config.UAV_RELOC_PERIOD

        self.logger.info("Total prop energy : {}".format(totalPropEnergy))

        return totalPropEnergy

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
                exisiting_cells[next_cell][1] = index
                exisiting_cells[next_cell][2] = new_dist
            else :
                next_state[index].power = 'off'

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

        return U2GState(self.uavPosition, sample_states, self.uavs, gmus, True)

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

    def sample_gmus(self):
        return [len(self.gmuStatus[gs]) for gs in self.gmuStatus]

    ''' ===================================================================  '''
    '''                 Implementation of abstract Model class               '''
    ''' ===================================================================  '''

    def reset_for_simulation(self):
        """
        The Simulator (Model) should be reset before each simulation
        :return:
        """

    def reset_for_epoch(self):
        self.actual_gmu_states = self.sample_gmus()
        self.logger.info("Actual GMU states : {}".format(self.actual_gmu_states))

    def update(self, sim_data):
        """
        Update the state of the simulator with sim_data
        :param sim_data:
        :return:
        """

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
        result.observation = self.make_observation(action, result.next_state)
        result.reward = self.make_reward(state, action, result.next_state)
        exit()
        result.is_terminal = self.is_terminal(result.next_state)

        return result


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

    def is_valid(self, state):
        """
        Returns true iff the given state is valid
        :param state:
        :return:
        """

