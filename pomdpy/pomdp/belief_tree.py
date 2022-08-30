from __future__ import absolute_import
from pomdpy.pomdp.belief_node import BeliefNode
from pomdpy.pomdp.belief_structure import BeliefStructure
from queue import PriorityQueue

class BeliefTree(BeliefStructure):
    """
    Contains the BeliefTree class, which represents an entire belief tree.
    *
    * Most of the work is done in the individual classes for the mappings and nodes; this class
    * simply owns a root node and handles pruning
    """
    def __init__(self, agent):
        super(BeliefTree, self).__init__()
        self.agent = agent
        self.root = None
        self.MaxActionPool = self.agent.model.MaxActionPool

    # --------- TREE MODIFICATION ------- #
    def reset(self):
        """
        Reset the tree
        :return:
        """
        self.prune_tree(self)
        self.root = BeliefNode(self.agent, None, None)
        return self.root

    def reset_root_data(self):
        """
        Completely resets the root data
        :return:
        """
        self.root.data = self.agent.model.create_root_historical_data(self.agent)

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
        self.reset_root_data()
        self.root.action_map = self.agent.action_pool.create_action_mapping(self.root)


    def prune_tree(self, bt):
        """
        Clears out a belief tree. This is difficult,
        because every node has a reference to its owner and child.
        """
        self.prune_node(bt.root)
        bt.root = None

    def prune_node(self, bn):
        """
        Remove node bn and all of its descendants from the belief tree
        :param bn:
        :return:
        """
        if bn is None:
            return

        # observation mapping entry
        bn.parent_entry = None

        # the action maps owner reference
        bn.action_map.owner = None

        action_mapping_entries = bn.action_map.get_child_entries()

        for entry in action_mapping_entries:
            # Action Node
            entry.child_node.parent_entry = None
            entry.map = None
            entry.child_node.observation_map.owner = None
            for observation_entry in list(entry.child_node.observation_map.child_map.values()):
                self.prune_node(observation_entry.child_node)
                observation_entry.map = None
                observation_entry.child_node = None
            entry.child_node.observation_map = None
            entry.child_node = None
        bn.action_map = self.agent.action_pool.create_action_mapping(bn)

    def prune_siblings(self, bn):

        if bn is None:
            return

        parent_belief = bn.get_parent_belief()

        if parent_belief is not None:
            # For all action entries with action nodes expanded out from the parent_belief (root of the belief tree)
            for action_mapping_entry in parent_belief.action_map.get_child_entries():
                # for every observation made
                for obs_mapping_entry in action_mapping_entry.child_node.observation_map.get_child_entries():
                    if not obs_mapping_entry.status_for_grabObs :
                        self.prune_node(obs_mapping_entry.child_node)


                # if Q.full() :
                #     Q.put(action_mapping_entry.mean_q_value)
                # else :
                #     print("Q size : {}".format(Q.qsize()))
                #     Min = Q.get()
                #     print("MIN : ", Min)
                #     print("new action Q : ", action_mapping_entry.mean_q_value)
                #     print("action_mapping_entry.deployment : ", action_mapping_entry.deployment)
                #     print("action.UAV_deployment : ", action.UAV_deployment)
                #
                #     if Min > action_mapping_entry.mean_q_value or action_mapping_entry.deployment == action.UAV_deployment:
                #         Q.put(Min)
                #     else :
                #         Q.put(action_mapping_entry.mean_q_value)


