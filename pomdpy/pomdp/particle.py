import random, math
import numpy as np
from collections import deque

class ParticlePool :
    def __init__(self, MaxParticle, solverType) :
        self.TotalParticle = 0
        self.MaxParticle = MaxParticle
        self.solverType = solverType
        self.createCurrentParticle = False

        if self.solverType == 0 or self.solverType == 2:
            self.particle = Particle(self.MaxParticle)
        elif self.solverType == 1 or self.solverType == 3:
            self.particle = ParticleOfOW(self.MaxParticle)

    def add_partcle(self, particle, type):
        added = self.particle.add_particle(particle, type)
        self.TotalParticle +=added

    def sample_particle(self):
        if self.solverType == 0 or self.solverType == 2 :
            return self.particle.random_particle(self.createCurrentParticle)
        else :
            return self.sample_particle_of_POMCPOW()

    def sample_particle_of_POMCPOW(self):
        return self.particle.select_maxNum_particle()

    def set_create_current_particle(self, _bool):
        self.createCurrentParticle = _bool

    def get_create_current_particle(self):
        return self.createCurrentParticle

    def get_num_total_particle(self):
        return self.TotalParticle

    def get_numbers_of_state(self, gmuState, cellIndex, GRID_W):
        num = 0
        diff = 0
        distribution = np.zeros(len(gmuState))
        for i in range(len(self.particle.state)) :
            s = self.particle.state[i].gmu_position
            if s == gmuState : num +=1

            distribution = distribution + s
            error = []
            for i, gmu in enumerate(self.particle.state[i].gmus):
                prediction_cell = gmu.get_cellCoordinate(GRID_W)

                error_x = int(cellIndex[i][0]) - prediction_cell[0]
                error_y = cellIndex[i][1] - prediction_cell[1]

                error.append(math.sqrt(math.pow(error_x, 2) + math.pow(error_y, 2)))

            diff += np.mean(error) / self.TotalParticle

        for i in range(len(self.particle.current_state)):
            s = self.particle.current_state[i].gmu_position
            if s == gmuState : num +=1

            distribution = distribution + s
            error = []
            for i, gmu in enumerate(self.particle.current_state[i].gmus) :
                prediction_cell = gmu.get_cellCoordinate(GRID_W)

                error_x = int(cellIndex[i][0]) - prediction_cell[0]
                error_y = cellIndex[i][1] - prediction_cell[1]

                error.append(math.sqrt(math.pow(error_x, 2) + math.pow(error_y, 2)))

            diff += np.mean(error) / self.TotalParticle

        distribution = distribution / self.TotalParticle
        print("current particle :", len(self.particle.current_state))
        print("particle :", len(self.particle.state))
        return num / self.TotalParticle, diff, distribution


class Particle :
    def __init__(self, MaxParticle):
        self.state = []
        self.current_state = []
        self.MaxParticle = MaxParticle

    def add_particle(self, state, type):
        if type == 0 :
            self.current_state.append(state)
            return 1

        elif type == 1 :
            if len(self.state) >= self.MaxParticle :
                return 0
            else :
                self.state.append(state)
                return 1


    def random_particle(self, createCurrentParticle):
        if createCurrentParticle :
            return random.choice(self.current_state)
        else :
            return random.choice(self.state)


class ParticleOfOW :
    def __init__(self, MaxParticle):
        self.state = []
        self.NumState = {}
        self.MaxParticle = MaxParticle
        self.MaxState = None
        self.MaxNum = 0
    def add_particle(self, state):
        if len(self.state) >= self.MaxParticle :
            return 0

        self.state.append(state)
        new_key = state.get_key()
        if new_key not in self.NumState :
            self.NumState[new_key] = 1
        else :
            self.NumState[new_key] +=1

        if self.NumState[new_key] > self.MaxNum :
            self.MaxNum = self.NumState[new_key]
            self.MaxState = state

        return 1

    def select_maxNum_particle(self):
        return self.MaxState
