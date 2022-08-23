import pandas as pd
from .trajectory_grid import TG
from .SL_object import MO, State, Trajectory
from .mobility_config import MConfig
from .utils import set_coordinate, get_cellCoordinate, get_state_transition_prob, create_random_position_in_cell, getGridIndex, get_id_of_gmu, add_noise_to_trajectory
import logging, random, copy

class SLModel :
    def __init__(self, NumOfMO, cellWidth, MAX_XGRID_N, exceptedID=-1):
        self.logger = logging.getLogger('POMDPy.SLModel')
        self.logger.setLevel("INFO")
        self.traj = {}

        self.cellWidth = cellWidth
        self.MAX_XGRID_N = MAX_XGRID_N
        self.sampling_interval = int((cellWidth/MConfig.velocity)//MConfig.interval)
        self.logger.debug("sampling interval : {}".format(self.sampling_interval))

        # read trajectories of MO
        # create TG
        # 4. TG AND UPDATE PROCESS
        self.tg = TG(self.logger)
        self.MOS = []
        self.NumObservedGMU = 0
        self.NumGMU = NumOfMO
        self.simulation_state = [None for _ in range(self.NumGMU)]
        self.initialize(NumOfMO, exceptedID)

        self.init_traj = copy.deepcopy(self.traj)
        self.init_tg = copy.deepcopy(self.tg)
        self.init_MOS = copy.deepcopy(self.MOS)



    def initialize(self, NumOfMO, exceptedID):
        for i in range(NumOfMO) :
            if i == exceptedID :
                continue
            file = "/home/kyungho/project/U2G_POMDPy/mobility/trajectory/MO" + str(i) +"_traj.csv"

            self.traj[i] = Trajectory(pd.read_csv(file))
            mo = MO("MO"+str(i))
            for j in range(0, MConfig.initialTrip*self.sampling_interval, self.sampling_interval) :
                self.update_trajectory(mo, i, j)

            self.logger.debug("GMU {}' trajectory updated until {} steps".format(i, self.traj[i].updated_time))
            self.MOS.append(mo)

        for i in range(len(self.MOS), len(self.MOS)+MConfig.Batch) :
            if i == exceptedID :
                continue
            file = "/home/kyungho/project/U2G_POMDPy/mobility/trajectory/MO" + str(i) +"_traj.csv"

            self.traj[i] = Trajectory(pd.read_csv(file))
            mo = MO("MO"+str(i))
            for j in range(0, MConfig.BatchInitialTrip*self.sampling_interval, self.sampling_interval) :
                self.update_trajectory(mo, i, j)

            self.logger.debug("GMU {}' trajectory updated until {} steps".format(i, self.traj[i].updated_time))
            self.MOS.append(mo)

    def reset(self, uavStatus):
        self.tg = copy.deepcopy(self.init_tg)
        self.MOS = copy.deepcopy(self.init_MOS)
        self.traj = copy.deepcopy(self.init_traj)
        self.reset_NumObservedGMU()

        for i in range(self.NumGMU) :
            self.create_gmu_for_simulation(i, uavStatus)

    def create_gmu_for_simulation(self, id, uavStatus):
        update_time = self.traj[id].updated_time + self.sampling_interval
        actual_next_loc = self.get_traj(id, update_time)

        if uavStatus[actual_next_loc[1][1]][actual_next_loc[1][0]] == 0:
            t0Loc = self.MOS[id].backward_traj[-1]
            ro = self.get_reference_objects(id, self.MOS[id].backward_traj)
            S = State()
            eta = 1
            k=1
            result = self.prediction_probabilistic_next_states(ro, MConfig.theta, t0Loc, S, eta, k)

            if result == [] :
                self.logger.error("There are no path of GMU [{}]".format(id))
                return -1, -1, -1, -1
            elif result[5] == True :
                self.logger.error("There are no next path of GMU [{}]".format(id))
                return -1, -1, -1, -1
            else:
                self.MOS[id].set_prediction(S, t0Loc, ro, eta, k)
            # create a random position corresponding its cell
            next_loc = create_random_position_in_cell(result[1][0], result[1][1], self.cellWidth)
            self.logger.debug("[{}]' trajectory is predictied : {}".format(self.MOS[id].id, next_loc))
            return self.MOS[id].id, next_loc, False, result
        else :
            update_time = self.traj[id].updated_time + self.sampling_interval
            self.update_trajectory(self.MOS[id], id, update_time)
            self.NumObservedGMU +=1
            self.logger.debug("[{}]' trajectory is updated : {}".format(self.MOS[id].id, self.MOS[id].get_location()))

            return self.MOS[id].id, self.MOS[id].get_location(), True, None


    def update_gmu_for_simulation(self, id, SL_parms, uavStatus):
        update_time = self.traj[id].updated_time + self.sampling_interval
        actual_next_loc = self.get_traj(id, update_time)

        if self.MOS[id].observed == False :
            self.MOS[id].set_prediction(SL_parms[0], SL_parms[1], SL_parms[2], SL_parms[3], SL_parms[4])

        if uavStatus[actual_next_loc[1][1]][actual_next_loc[1][0]] == 0 :
            if self.MOS[id].observed == True :
                t0Loc = self.MOS[id].backward_traj[-1]
                ro = self.get_reference_objects(id, self.MOS[id].backward_traj)
                S = State()
                eta = 1
                k=1
                result = self.prediction_probabilistic_next_states(ro, MConfig.theta, t0Loc, S, eta, k)

                if result == [] :
                    self.logger.error("There is no path of GMU [{}]".format(id))
                elif result[5] == True :
                    self.logger.error("There are no next path of GMU [{}]".format(id))
                else :
                    self.MOS[id].set_prediction(S, t0Loc, ro, eta, k)
            else :
                try :
                    S, t0Loc, RO, eta, k = self.MOS[id].get_mobility_model()
                except :
                    self.logger.error("TEST : {}".format(SL_parms))
                    exit()
                result = self.prediction_probabilistic_next_states(RO, MConfig.theta, t0Loc, S, eta, k)

                if result == [] :
                    self.logger.error("There is no path of GMU [{}]".format(id))
                elif result[5] == True :
                    self.logger.error("There are no next path of GMU [{}]".format(id))
        else :
            self.NumObservedGMU +=1
            if self.MOS[id].observed == False:
                for _ in range(self.MOS[id].k) :
                    updated_time = self.traj[id].updated_time + self.sampling_interval
                    self.update_trajectory(self.MOS[id], id, updated_time)

                self.MOS[id].reset_prediction()

            else :
                self.update_trajectory(self.MOS[id], id, update_time)

            self.logger.debug("[{}]' trajectory is updated : {}".format(self.MOS[id].id, self.MOS[id].get_location()))



    def get_init_gmu_locIndex(self, id):
        if self.MOS[id].observed == True :
            xC, yC = self.MOS[id].get_cell_location()
            index = getGridIndex(xC, yC, self.MAX_XGRID_N)

            return index, self.MOS[id].get_location(), True, None
        else :
            S, t0Loc, ro, eta, k = self.MOS[id].get_mobility_model()

            result = self.prediction_probabilistic_next_states(ro, MConfig.theta, t0Loc, S, eta, k)

            if result == []:
                self.logger.error("There are no path of GMU [{}], {}".format(id, k))
                return -1, -1, -1, -1
            elif result[5] == True:
                self.logger.error("There is no next path of GMU [{}] at k {}".format(id, k))
                self.logger.error("result : {}".format(result))
                return -1, -1, -1, -1
            # create a random position corresponding its cell
            index = getGridIndex(result[1][0], result[1][1], self.MAX_XGRID_N)

            # create a random position corresponding its cell
            next_loc = create_random_position_in_cell(result[1][0], result[1][1], self.cellWidth)

            return index, next_loc, False, result


    def get_trajectory_for_simulation(self, id):
        S, t0Loc, ro, eta, k = self.simulation_state[id]
        result = self.prediction_probabilistic_next_states(ro, MConfig.theta, t0Loc, S, eta, k)
        self.simulation_state[id] = result[0:5]

        if result == [] :
            self.logger.info("ID :{}', There are no trajectories in having RO : {}".format(id, k))
            if k == 1:
                self.logger.info("t0Loc : {}, RO : {}".format(t0Loc, ro))

            return -1, -1, True, -1

        # create a random position corresponding its cell
        index = getGridIndex(result[1][0], result[1][1], self.MAX_XGRID_N)

        # create a random position corresponding its cell
        next_loc = create_random_position_in_cell(result[1][0], result[1][1], self.cellWidth)

        overed = self.check_gmu_trajectory_overed(id)
        tree_depth_overd = self.check_tree_depth(result[4])
        terminal = overed or tree_depth_overd or result[5]

        return index, next_loc, terminal, result


    def get_traj(self, id, update_time):
        _coordinate = self.traj[id].get_trajectory(update_time)
        _coordinate = set_coordinate(_coordinate[0], _coordinate[1], MConfig.x0, MConfig.y0)
        coordinate = get_cellCoordinate(_coordinate[0], _coordinate[1], self.cellWidth)

        return _coordinate,coordinate

    def update_trajectory(self, mo, id, update_time):
        _coordinate = self.traj[id].get_trajectory(update_time)
        _coordinate = set_coordinate(_coordinate[0], _coordinate[1], MConfig.x0, MConfig.y0)
        coordinate = get_cellCoordinate(_coordinate[0], _coordinate[1], self.cellWidth)

        if coordinate == -1:
            self.logger.error("over coordinate :", _coordinate)
            exit()

        self.tg.add_new_trajectory(coordinate, mo.get_cell_location(), mo.id, mo.current_t)
        mo.update_location(_coordinate, coordinate)
        self.traj[id].update_time(update_time)

    def test_update_trajectory(self, id, T):
        file = "/home/kyungho/project/POMDPy/mobility/trajectory/MO" + str(id) + "_traj.csv"

        self.traj[id] = Trajectory(pd.read_csv(file))
        mo = MO("MO" + str(id))
        for j in range(0, T * self.sampling_interval, self.sampling_interval):
            self.update_trajectory(mo, id, j)

        self.logger.debug("GMU {}' trajectory updated until {} steps".format(id, self.traj[id].updated_time))
        self.MOS.append(mo)

    def get_reference_objects(self, id, backward_traj):
        # find reference objects of mo0
        # 5. LOOKUP PROCESS

        RO = self.tg.lookup(self.MOS[id].id, backward_traj, self.MOS, self.NumGMU)
        self.logger.debug("Selected RO : {}".format(RO))

        if RO == [] :
            self.logger.info("There are no RO {} in {}".format(RO, id))
            new_backward_traj = add_noise_to_trajectory(list(backward_traj))
            get_id_of_gmu(id, new_backward_traj)

        return RO

    # Probabilistic Path Prediction Alogorithm
    # 6. PF AND CONSTRUCTION PROCESS
    def prediction_probabilistic_next_states(self, RO, theta, t0Loc, _S, eta, k) :
        S = copy.deepcopy(_S)
        Length_init_RO = len(RO)
        if len(RO)<= 0 :
            return []
        else :
            previousRO = []
            totalDensity = 0
            self.logger.debug("K : {}========================================================================".format(k))
            if eta > theta and k < self.get_max_path() :
                for ro in RO :
                    if ro not in self.tg.leafCells[t0Loc].trajectories :
                        self.logger.error("Current k: {}".format(k))
                        self.logger.error("not path : {}".format(ro))
                        self.logger.error("not t0Loc : {}".format(t0Loc))

                        continue

                    x, y = self.tg.leafCells[t0Loc].trajectories[ro].get_loc()
                    cellIndex = getGridIndex(x, y, self.MAX_XGRID_N)

                    next_ro = ro[0], ro[1] + 1
                    self.logger.debug("{}' next loc : {}, {} -> cellIndex : {}".format(next_ro, x, y, cellIndex))

                    previousRO.append(ro)
                    ros, createdNewState = S.update(k, (x,y), cellIndex, next_ro)

                    if createdNewState:
                        density = self.tg.leafCells[(x, y)].get_density()
                        totalDensity += density

            if totalDensity == 0:
                self.logger.error("TEST : {}".format(len(RO)))
                self.logger.error("RO : {}".format(S.states.keys()))
                return []

            self.logger.debug("S[{}] - RO : {}".format(k, ros))
            self.logger.debug("totalDensity : {}".format(totalDensity))
            self.logger.debug("previousRO : {}".format(previousRO))

            new_t0Loc, new_RO , new_eta = self.calcurate_eta(eta, Length_init_RO, previousRO, totalDensity, S, k)
            is_terminal = self.check_having_trajectory_RO(new_t0Loc, new_RO, k)
            return [S, new_t0Loc, new_RO , new_eta, k+1, is_terminal]

    def prediction_probabilistic_path(self, RO, theta, id) :
        Length_init_RO = len(RO)
        if len(RO)<= 0 :
            return []
        else :
            eta = 1 # probability that the path ends with predicted state
            selectedPath = []
            k = 1
            S = State()  # state
            t0Loc = self.MOS[id].backward_traj[-1]
            while eta > theta and len(selectedPath) < self.get_max_path():
                totalDensity = 0
                previousRO = []
                selectedPath.append(t0Loc)
                self.logger.debug("K : {}========================================================================".format(k))
                for ro in RO :
                    if ro not in self.tg.leafCells[t0Loc].trajectories :
                        self.logger.debug("not path : {}".format(ro))
                        continue
                    x, y = self.tg.leafCells[t0Loc].trajectories[ro].get_loc()
                    cellIndex = getGridIndex(x, y, self.MAX_XGRID_N)

                    next_ro = ro[0], ro[1] +1
                    self.logger.debug("{}' next loc : {}, {} -> cellIndex : {}".format(next_ro, x, y, cellIndex))

                    previousRO.append(ro)
                    ros, createdNewState = S.update(k, (x,y), cellIndex, next_ro)

                    if createdNewState:
                        density = self.tg.leafCells[(x,y)].get_density()
                        totalDensity +=density

                if totalDensity == 0 :
                    break

                self.logger.debug("S[{}] - RO : {}".format(k, ros))
                self.logger.debug("totalDensity : {}".format(totalDensity))
                self.logger.debug("previousRO : {}".format(previousRO))

                t0Loc, RO , eta = self.calcurate_maxEta(eta,Length_init_RO, previousRO, totalDensity, S, k)

                k+=1

            self.logger.debug("selected Path : {}".format(selectedPath))

        return selectedPath

    # calcurate probability for next state
    def calcurate_eta(self, eta, Length_init_RO, previousRO, totalDensity, S, k) :
        max_states = []
        etas = []
        next_ROs = []
        for next_s, v in list(S.get_key_value_of_k(k)) :
            union, intersection, state_transition_prob = get_state_transition_prob(previousRO, v.currentRO)

            self.logger.debug("state : {}---------------------------------------------------------------------------".format(next_s))
            self.logger.debug("currentRO : {}".format(v.currentRO))
            self.logger.debug("union : {}, intersection : {}".format(union, intersection))
            self.logger.debug("state_transition_prob : {}".format(state_transition_prob))

            # likelihood Function
            prior_prob = (totalDensity - self.tg.leafCells[next_s].get_density()) / totalDensity +0.01
            distribution_RO = len(v.currentRO) / Length_init_RO
            # RO_prob = 1 / len(S.get_key_value_of_k(k))
            RO_prob = MConfig.RO_prob

            likelihood = distribution_RO*RO_prob / prior_prob
            # likelihood = 1
            new_eta = eta * state_transition_prob * likelihood
            S.states[k][next_s].probability = new_eta
            self.logger.debug("Density : {}/{}".format(self.tg.leafCells[next_s].get_density(), totalDensity))
            self.logger.debug("prior_prob : {}".format(prior_prob))
            self.logger.debug("distribution_RO : {}".format(distribution_RO))
            self.logger.debug("RO_prob : {}".format(RO_prob))
            self.logger.debug("likelihood : {}".format(likelihood))
            self.logger.debug("new_eta : {}".format(new_eta))

            max_states.append(next_s)
            next_ROs.append(v.currentRO)
            etas.append(new_eta)
            # if max <= new_eta :
            #     max = new_eta
            #     max_state = next_s
            #     next_RO = v.currentRO

        selected_index = max_states.index(random.choices(max_states, etas)[0])

        self.logger.debug("selected State : {}".format(max_states[selected_index]))
        self.logger.debug("selected eta : {}".format(etas[selected_index]))
        self.logger.debug("selected RO : {}".format(next_ROs[selected_index]))

        return [max_states[selected_index], next_ROs[selected_index], etas[selected_index]]

    def calcurate_maxEta(self, eta, Length_init_RO, previousRO, totalDensity, S, k):
        max_states = []
        etas = []
        next_ROs = []
        for next_s, v in list(S.get_key_value_of_k(k)):
            union, intersection, state_transition_prob = get_state_transition_prob(previousRO, v.currentRO)

            self.logger.debug(
                "state : {}---------------------------------------------------------------------------".format(next_s))
            self.logger.debug("currentRO : {}".format(v.currentRO))
            self.logger.debug("union : {}, intersection : {}".format(union, intersection))
            self.logger.debug("state_transition_prob : {}".format(state_transition_prob))

            # likelihood Function
            prior_prob = (totalDensity - self.tg.leafCells[next_s].get_density()) / totalDensity
            if prior_prob == 0 :
                prior_prob = 1
            distribution_RO = len(v.currentRO) / Length_init_RO
            # RO_prob = 1 / len(S.get_key_value_of_k(k))
            RO_prob = MConfig.RO_prob

            likelihood = distribution_RO * RO_prob / prior_prob
            # likelihood = 1
            new_eta = eta * state_transition_prob * likelihood
            S.states[k][next_s].probability = new_eta
            self.logger.debug("Density : {}/{}".format(self.tg.leafCells[next_s].get_density(), totalDensity))
            self.logger.debug("prior_prob : {}".format(prior_prob))
            self.logger.debug("distribution_RO : {}".format(distribution_RO))
            self.logger.debug("RO_prob : {}".format(RO_prob))
            self.logger.debug("likelihood : {}".format(likelihood))
            self.logger.debug("new_eta : {}".format(new_eta))

            max_states.append(next_s)
            next_ROs.append(v.currentRO)
            etas.append(new_eta)
            # if max <= new_eta :
            #     max = new_eta
            #     max_state = next_s
            #     next_RO = v.currentRO

        Max = max(etas)
        selected_index = etas.index(Max)
        self.logger.debug("selected State : {}".format(max_states[selected_index]))
        self.logger.debug("selected eta : {}".format(etas[selected_index]))
        self.logger.debug("selected RO : {}".format(next_ROs[selected_index]))

        return max_states[selected_index], next_ROs[selected_index], etas[selected_index]

    def get_max_path(self):
        return MConfig.MaxPath +1

    def get_num_observed_GMU(self):
        return self.NumObservedGMU

    def get_gmu_env(self, Row, Column):
        row = [0 for _ in range(Row)]
        envMap = [row[:] for _ in range(Column)]
        for i in range(self.NumGMU) :
            x, y = self.MOS[i].get_cell_location()
            envMap[y][x] +=1

        return envMap

    def get_gmu_position(self, MAX_GRID_INDEX):
        gmu_position = [0 for _ in range(MAX_GRID_INDEX)]

        for i in range(self.NumGMU) :
            x, y = self.MOS[i].get_cell_location()
            cellIndex = getGridIndex(x, y, self.MAX_XGRID_N)

            gmu_position[cellIndex] +=1

        return gmu_position


    def check_gmu_trajectory_overed(self, id):
        if self.MOS[id].get_current_time() + 1 > MConfig.MaxTrajectory:
            return True

        return False

    def check_tree_depth(self, k):
        if k + 1 > self.get_max_path():
            return True

        return False

    def check_having_trajectory_RO(self, loc, RO, k):
        for ro in RO :
            if ro not in self.tg.leafCells[loc].trajectories:
                id = get_id_of_gmu(ro[0])
                self.logger.error("don't have more trajectory : {} in {} at {}".format(ro, loc, k))
                self.logger.error("t : {}".format(self.MOS[id].get_current_time()))
                self.logger.error("t2 : {}".format(self.MOS[id].test))
                self.logger.error("t3 : {}".format(self.MOS[id].test2))

                for history in self.MOS[id].test :
                    for new_ro in self.tg.leafCells[history].trajectories :
                        if new_ro[0] == ro[0] and new_ro[1]>240:
                            self.logger.error("t4 : {}/{}".format(new_ro, history))
                # self.logger.error("t3 : {}".format(self.tg[))
                return True

        return False

    def reset_NumObservedGMU(self):
        self.NumObservedGMU = 0

    def reset_simulation_state(self, id, SL_parms):
        if SL_parms == None :
            t0Loc = self.MOS[id].backward_traj[-1]
            ro = self.get_reference_objects(id, self.MOS[id].backward_traj)
            S = State()
            eta = 1
            k = 1
            self.simulation_state[id] = [S, t0Loc, ro, eta, k]
        else :
            self.simulation_state[id] = SL_parms
