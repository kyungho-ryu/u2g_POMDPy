import random
from collections import deque

class ParticlePool :
    def __init__(self, MaxParticle, solverType) :
        self.TotalParticle = 0
        self.NumParticle = {}
        self.particle = {}
        self.MaxParticle = MaxParticle
        self.solverType = solverType


    def add_partcle(self, particle, prior_state):
        if prior_state not in self.particle:
            if self.solverType == 0 or self.solverType == 2 :
                self.particle[prior_state] = Particle(self.MaxParticle)
            elif self.solverType == 1 or self.solverType == 3 :
                self.particle[prior_state] = ParticleOfOW()

            self.NumParticle[prior_state] = 0

        added = self.particle[prior_state].add_particle(particle)
        self.NumParticle[prior_state] +=added
        self.TotalParticle +=added


    def sample_particle(self, prior_state):
        if self.solverType == 0 or self.solverType == 2 :
            if prior_state not in self.particle :
                particle = random.choice(list(self.particle.values()))
                return particle.random_particle()
            elif self.NumParticle[prior_state] < self.MaxParticle:
                particle = random.choice(list(self.particle.values()))
                return particle.random_particle()
            else :
                return self.particle[prior_state].random_particle()

        else :
            return self.sample_particle_of_POMCPOW(prior_state)

    def sample_particle_of_POMCPOW(self, prior_state):
        if prior_state not in self.particle :
            particle = random.choice(list(self.particle.values()))
            return particle.random_particle()
        else :
            return self.particle[prior_state].select_maxNum_particle()


    def get_num_leftParticle_of_priorState(self, prior_state):
        if prior_state not in self.NumParticle :
            return self.MaxParticle
        else :
            return self.MaxParticle - self.NumParticle[prior_state]

    def get_num_total_particle(self):
        return self.TotalParticle

class Particle :
    def __init__(self, MaxParticle):
        self.state = []
        self.MaxParticle = MaxParticle

    def add_particle(self, state):
        if len(self.state) >= self.MaxParticle :
            return 0
        else :
            self.state.append(state)
            return 1

    def random_particle(self):
        return random.choice(self.state)


class ParticleOfOW :
    def __init__(self):
        self.state = []
        self.NumState = {}
        self.MaxState = None
        self.MaxNum = 0
    def add_particle(self, state):
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
