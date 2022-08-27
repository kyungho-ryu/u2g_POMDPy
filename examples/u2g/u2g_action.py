from __future__ import print_function
from builtins import object
from pomdpy.discrete_mapping_pomdp import DiscreteU2GAction
import hashlib

class U2GAction(DiscreteU2GAction):
    """
    - Action class
    - UAV deployment
    -Handles pretty printing
    """

    def __init__(self, deployment):
        super(U2GAction, self).__init__(deployment)

    def copy(self):
        return U2GAction(self.UAV_deployment)

    def print_action(self):
        print(self.UAV_deployment)

    def to_string(self):
        return str(self.UAV_deployment)

    def distance_to(self, other_point):
        pass

    def get_key(self):
        return hashlib.sha256(self.to_string().encode()).hexdigest()

    # def __del__(self):
    #     print(f'destroy {id(self)}')