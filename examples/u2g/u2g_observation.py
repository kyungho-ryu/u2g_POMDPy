from __future__ import print_function
from builtins import str
from pomdpy.discrete_mapping_pomdp import DiscreteU2GObservation
import hashlib

class U2GObservation(DiscreteU2GObservation):
    """
    Default behavior is for the rock observation to say that the rock is empty
    """
    def __init__(self, observed_gmu_status):
        super(U2GObservation, self).__init__(observed_gmu_status)

    def copy(self):
        return U2GObservation(self.observed_gmu_status)

    def check_dissimilarity(self, other_u2g_observation):
        dissimilarity = 0
        for i in range(len(self.observed_gmu_status)) :
            if self.observed_gmu_status[i] != other_u2g_observation.observed_gmu_status[i] :
                if self.observed_gmu_status[i] > 0 and other_u2g_observation.observed_gmu_status[i] > 0 :
                    continue
                else:
                    dissimilarity +=1

        return dissimilarity/len(self.observed_gmu_status)

    def __eq__(self, other_u2g_observation):
        key = self.get_key(self.observed_gmu_status)
        return key == other_u2g_observation
    def get_key(self, obs):
        return hashlib.sha256(str(obs).encode()).hexdigest()

    def __hash__(self):
        # return (False, True)[self.is_good]
        pass

    def print_observation(self):
            print("observed GMU : ", self.observed_gmu_status)

    def to_string(self):
        return str(self.observed_gmu_status)
