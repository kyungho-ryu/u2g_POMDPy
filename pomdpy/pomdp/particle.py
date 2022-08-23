import random
from collections import deque


MinParticle = 100

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
        if self.NumParticle[prior_state] < MinParticle:
            particle = random.choice(list(self.particle.values()))
            return particle.random_particle()
        else :
            return self.particle[prior_state].random_particle()


class Particle :
    def __init__(self, ):
        self.state = deque(maxlen=100)

    def add_particle(self, state):
        self.state.append(state)

    def random_particle(self):
        return random.choice(self.state)

