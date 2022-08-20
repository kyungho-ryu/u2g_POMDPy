from __future__ import absolute_import
from pomdpy.pomdp.belief_mapping_node import BeliefMappingNode
from pomdpy.pomdp.belief_structure import BeliefStructure
from pomdpy.util.mapping import get_key

class BeliefMapping(BeliefStructure):
    """
    Contains the BeliefTree class, which represents an entire belief tree.
    *
    * Most of the work is done in the individual classes for the mappings and nodes; this class
    * simply owns a root node and handles pruning
    """
    def __init__(self, agent):
        super(BeliefMapping, self).__init__()
        self.agent = agent
        self.beliefMap = dict()

    # --------- TREE MODIFICATION ------- #
    def reset(self, observation):
        """
        Reset the mapping
        :return:
        """
        # root -> beliefMap
        key = get_key(observation.observed_gmu_status)
        self.beliefMap[key] = BeliefMappingNode(self.agent, self)

        return self.beliefMap


    def reset_data(self, root_data=None):
        """
        Keeps information from the provided root node
        :return:
        """
        if root_data is not None:
            self.root.data.reset(root_data)
        else:
            self.root.data.reset()

    def initialize(self, init_value=None):
        key = get_key(init_value.observed_gmu_status)
        action_map = self.agent.action_pool.create_action_mapping(self.beliefMap[key])

        self.beliefMap[key].action_map = action_map

    def create_belief_node(self, obs, beliefNode):
        key = get_key(obs.observed_gmu_status)
        self.beliefMap[key] = beliefNode

    def add_particle(self, obs, particle):
        key = get_key(obs.observed_gmu_status)

        self.beliefMap[key].state_particles.append(particle)

    def get_belief_node(self, obs):
        key = get_key(obs.observed_gmu_status)

        if key in self.beliefMap :
            return self.beliefMap[key]
        else :
            return None