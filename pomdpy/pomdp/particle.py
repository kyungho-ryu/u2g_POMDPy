import random
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

    def add_partcle(self, particle):
        added = self.particle.add_particle(particle)

        self.TotalParticle +=added

    def sample_particle(self):
        if self.solverType == 0 or self.solverType == 2 :
            return self.particle.random_particle()
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
