import random
from collections import deque


MinParticle = 10

class ParticlePool :
    def __init__(self) :
        self.TotalParticle = 0
        self.NumParticle = {}
        self.particle = {}


    def add_partcle(self, particle, prior_state):
        if prior_state not in self.particle:
            self.particle[prior_state] = Particle()
            self.NumParticle[prior_state] = 0

        self.particle[prior_state].add_particle(particle)
        self.NumParticle[prior_state] +=1
        self.TotalParticle +=1


    def sample_particle(self, prior_state):
        if prior_state not in self.particle :
            particle = random.choice(list(self.particle.values()))
            return particle.random_particle()
        elif self.NumParticle[prior_state] < MinParticle:
            particle = random.choice(list(self.particle.values()))
            return particle.random_particle()
        else :
            return self.particle[prior_state].random_particle()


    def get_num_total_particle(self):
        return self.TotalParticle
class Particle :
    def __init__(self, ):
        self.state = []

    def add_particle(self, state):
        if len(self.state) >= 10 :
            self.state.pop(0)
        self.state.append(state)

    def random_particle(self):
        return random.choice(self.state)

