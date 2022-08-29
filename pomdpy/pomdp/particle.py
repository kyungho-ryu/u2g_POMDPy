import random
from collections import deque


class ParticlePool :
    def __init__(self, MaxParticle) :
        self.TotalParticle = 0
        self.NumParticle = {}
        self.particle = {}
        self.MaxParticle = MaxParticle


    def add_partcle(self, particle, prior_state):
        if prior_state not in self.particle:
            self.particle[prior_state] = Particle(self.MaxParticle)
            self.NumParticle[prior_state] = 0
        # if len(self.particle.keys()) > MaxParticle :
        #     key = list(self.particle.keys())[0]
        #     self.TotalParticle -= self.NumParticle[key]
        #
        #     self.particle.pop(key)
        #     self.NumParticle.pop(key)


        added = self.particle[prior_state].add_particle(particle)
        self.NumParticle[prior_state] +=added
        self.TotalParticle +=added


    def sample_particle(self, prior_state):
        if prior_state not in self.particle :
            particle = random.choice(list(self.particle.values()))
            return particle.random_particle()
        elif self.NumParticle[prior_state] < self.MaxParticle:
            particle = random.choice(list(self.particle.values()))
            return particle.random_particle()
        else :
            return self.particle[prior_state].random_particle()


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
