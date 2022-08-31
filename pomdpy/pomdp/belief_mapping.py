from __future__ import absolute_import

import gc

from pomdpy.pomdp.belief_mapping_node import BeliefMappingNode
from pomdpy.pomdp.belief_structure import BeliefStructure
from pomdpy.util.mapping import get_key

import copy
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
        self.beliefMap[key] = BeliefMappingNode(self.agent)

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

    def copy_belief_node(self, obs, new_obs):
        key = get_key(obs.observed_gmu_status)
        new_key = get_key(new_obs.observed_gmu_status)

        self.beliefMap[new_key] = self.beliefMap[key].deep_copy()

    def add_particle(self, obs, particle):
        key = get_key(obs.observed_gmu_status)

        self.beliefMap[key].add_particle(particle)

    def get_belief_node(self, obs):
        key = get_key(obs.observed_gmu_status)

        if key in self.beliefMap :
            return self.beliefMap[key]
        else :
            return None

    def prune_siblings(self, bn):
        """
        Prune all of the sibling nodes of the provided belief node, leaving the parents
        and ancestors of bn intact
        :param bn:
        :return:
        """
        if bn is None:
            return

        if bn is not None:
            # For all action entries with action nodes expanded out from the parent_belief (root of the belief tree)
            for action, action_mapping_entry in list(bn.action_map.entries.items()):
                bn.action_map.entries[action] = None
                bn.action_map.entries.pop(action)
            print(bn.action_map.entries)
        print(len(bn.action_map.entries))
