import pandas as pd
from .trajectory_grid import TG
from .SL_object import MO, State, Trajectory, GMU
from .mobility_config import MConfig
from .utils import set_coordinate, get_cellCoordinate, get_cellIndex, get_state_transition_prob, create_random_position_in_cell, getGridIndex
import logging, random, copy

class SLModel :
    def __init__(self, NumOfMO, cellWidth, USER_DEMAND, MAX_XGRID_N, exceptedID=-1):
        self.logger = logging.getLogger('POMDPy.SLModel')
        self.logger.setLevel("INFO")
        self.traj = {}
        # GMU' current location

        self.cellWidth = cellWidth
        self.MAX_XGRID_N = MAX_XGRID_N
        self.sampling_interval = int((cellWidth/MConfig.velocity)//MConfig.interval)
        self.logger.debug("sampling interval : {}".format(self.sampling_interval))

        # read trajectories of MO
        # create TG
        # 4. TG AND UPDATE PROCESS
        self.tg = TG(self.logger)
        self.MOS = []
        for i in range(NumOfMO) :
            if i == exceptedID :
                continue
            file = "/home/kyungho/project/U2G_POMDPy/mobility/trajectory/MO" + str(i) +"_traj.csv"

            self.traj[i] = Trajectory(pd.read_csv(file))
            mo = GMU("MO"+str(i), USER_DEMAND)
            for j in range(0, MConfig.initialTrip*self.sampling_interval, self.sampling_interval) :
                self.update_trajectory(mo, i, j)

            self.logger.debug("GMU {}' trajectory updated until {} steps".format(i, self.traj[i].updated_time))
            self.MOS.append(mo)

        for i in range(len(self.MOS), len(self.MOS)+MConfig.Batch) :
            if i == exceptedID :
                continue
            file = "/home/kyungho/project/U2G_POMDPy/mobility/trajectory/MO" + str(i) +"_traj.csv"

            self.traj[i] = Trajectory(pd.read_csv(file))
            mo = GMU("MO"+str(i), USER_DEMAND)
            for j in range(0, MConfig.BatchInitialTrip*self.sampling_interval, self.sampling_interval) :
                self.update_trajectory(mo, i, j)

            self.logger.debug("GMU {}' trajectory updated until {} steps".format(i, self.traj[i].updated_time))
            self.MOS.append(mo)


    def update_gmu_location(self, id, uavStatus):
        cellCoordinate = self.MOS[id].get_current_loc()

        if uavStatus[cellCoordinate[1]][cellCoordinate[0]] == 0 :
            if self.MOS[id].observed == True :
                t0Loc = self.MOS[id].backward_traj[-1]
                ro = self.get_reference_objects(id, self.MOS[id].backward_traj)
                S = State()
                eta = 1
                k=1
                result = self.prediction_probabilistic_next_states(ro, MConfig.theta, t0Loc, S, eta, k)

                if result == [] :
                    self.logger.error("There is no path of GMU [{}]".format(id))
                    return -1, -1, -1, -1
                elif result[5] == True :
                    pass
                else :
                    self.MOS[id].set_prediction(result[0], result[2], result[3], result[4])
            else :
                S, RO, eta, k, t0Loc = self.MOS[id].get_mobility_model()
                result = self.prediction_probabilistic_next_states(RO, MConfig.theta, t0Loc, S, eta, k)

                if result == [] :
                    self.logger.error("There is no path of GMU [{}]".format(id))
                    return -1, -1, -1, -1
                elif result[5] == True :
                    pass
                else :
                    self.MOS[id].set_prediction(result[0], result[2], result[3], result[4])

            # create a random position corresponding its cell
            next_loc = create_random_position_in_cell(result[1][0], result[1][1], self.cellWidth)

            self.logger.debug("[{}]' trajectory is predictied : {}".format(self.MOS[id].id, next_loc))
            return self.MOS[id].id, next_loc, False, result
        else :
            if self.MOS[id].observed == False:
                for _ in range(self.MOS[id].k) :
                    updated_time = self.traj[id].updated_time + self.sampling_interval
                    self.update_trajectory(self.MOS[id], id, updated_time)

                self.MOS[id].reset_prediction()

            else :
                update_time = self.traj[id].updated_time + self.sampling_interval
                self.update_trajectory(self.MOS[id], id, update_time)

            self.logger.debug("[{}]' trajectory is updated : {}".format(self.MOS[id].id, self.MOS[id].get_location()))

            return self.MOS[id].id, self.MOS[id].get_location(), True, None

    def get_init_gmu_locIndex(self, id):
        if self.MOS[id].observed == True :
            xC, yC = self.MOS[id].get_current_loc()
            index = getGridIndex(xC, yC, self.MAX_XGRID_N)

            return index, self.MOS[id].get_location(), True, None
        else :
            t0Loc = self.MOS[id].backward_traj[-1]
            ro = self.get_reference_objects(id, self.MOS[id].backward_traj)
            S = State()
            eta = 1
            k = 1
            result = self.prediction_probabilistic_next_states(ro, MConfig.theta, t0Loc, S, eta, k)

            if result == []:
                self.logger.error("There is no path of GMU [{}]".format(id))
                return -1, -1, -1, -1
            elif result[5] == True:
                pass
            # create a random position corresponding its cell
            index = getGridIndex(result[1][0], result[1][1], self.MAX_XGRID_N)

            # create a random position corresponding its cell
            next_loc = create_random_position_in_cell(result[1][0], result[1][1], self.cellWidth)

            return index, next_loc, False, result

    def get_gmu_locIndex(self, id, k, *args):
        if k==1 :
            t0Loc = self.MOS[id].backward_traj[-1]
            ro = self.get_reference_objects(id, self.MOS[id].backward_traj)
            S = State()
            eta = 1
            k = 1
            result = self.prediction_probabilistic_next_states(ro, MConfig.theta, t0Loc, S, eta, k)

        else :
            result = self.prediction_probabilistic_next_states(
                args[0][2], MConfig.theta, args[0][0], args[0][1], args[0][3], args[0][4], id)

        if result == [] :
            self.logger.info("ID :{}', There are no trajectories in having RO : {}".format(id, k))
            return -1, -1, -1, -1

        # create a random position corresponding its cell
        index = getGridIndex(result[1][0], result[1][1], self.MAX_XGRID_N)

        # create a random position corresponding its cell
        next_loc = create_random_position_in_cell(result[1][0], result[1][1], self.cellWidth)
        return index, next_loc, False, result


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
        #
        # if id == 3 :
        #     print("test", mo.current_t, mo.id, mo.current_loc, coordinate)

        self.tg.add_new_trajectory(coordinate, mo.current_loc, mo.id, mo.current_t)
        mo.update_location(_coordinate, coordinate)
        self.traj[id].update_time(update_time)


    def test(self, ro, own_id, t0Loc):
        id = int(ro[0][-1])
        update_time = int(ro[1])
        print(self.MOS[id].current_t)

        prior_ro = ro[0], ro[1] -1
        print("prior ro", prior_ro)
        # print(self.get_reference_objects(own_id, self.MOS[own_id].backward_traj))
        x, y = self.tg.leafCells[t0Loc].trajectories[prior_ro].get_loc()
        print("x, y", x, y)

        if ro in self.tg.leafCells[(x, y)].trajectories :
            print("??")
        else :
            print(self.tg.leafCells[(x, y)].trajectories.keys())
            print("XX")
        print(self.tg.leafCells[t0Loc].trajectories.keys())
        exit()
    def test_update_trajectory(self, id, T):
        file = "/home/kyungho/project/POMDPy/mobility/trajectory/MO" + str(id) + "_traj.csv"

        self.traj[id] = Trajectory(pd.read_csv(file))
        mo = GMU("MO" + str(id), 15)
        for j in range(0, T * self.sampling_interval, self.sampling_interval):
            self.update_trajectory(mo, id, j)

        self.logger.debug("GMU {}' trajectory updated until {} steps".format(id, self.traj[id].updated_time))
        self.MOS.append(mo)

    def get_reference_objects(self, id, backward_traj):
        # find reference objects of mo0
        # 5. LOOKUP PROCESS
        RO = self.tg.lookup(self.MOS[id].id, backward_traj, self.MOS)
        self.logger.debug("Selected RO : {}".format(RO))

        return RO

    # Probabilistic Path Prediction Alogorithm
    # 6. PF AND CONSTRUCTION PROCESS
    def prediction_probabilistic_next_states(self, RO, theta, t0Loc, _S, eta, k, id=-1) :
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
                        self.logger.error("{}, {}".format(id, self.MOS[id].backward_traj[-1]))
                        self.logger.error("Current k: {}".format(k))
                        self.logger.error("not path : {}".format(ro))
                        self.logger.error("not t0Loc : {}".format(t0Loc))
                        self.test(ro, id,self.MOS[id].backward_traj[-1])

                        continue

                    x, y = self.tg.leafCells[t0Loc].trajectories[ro].get_loc()
                    scale = (MConfig.xE - MConfig.x0) // self.cellWidth
                    cellIndex = get_cellIndex(x, y, scale)

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
            is_terminal = self.check_having_trajectory_RO(new_t0Loc, new_RO)
            return [S, new_t0Loc, new_RO , new_eta, k+1, is_terminal]

    def check_having_trajectory_RO(self, loc, RO):
        for ro in RO :
            if ro not in self.tg.leafCells[loc].trajectories:
                return True

        return False

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
                    scale = (MConfig.xE - MConfig.x0) // self.cellWidth
                    cellIndex = get_cellIndex(x, y, scale)

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