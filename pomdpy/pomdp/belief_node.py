from builtins import object
import random
from pomdpy.pomdp.particle import ParticlePool

class BeliefNode(object):
    """
    Represents a single node in a belief tree.
    *
    * The key functionality is a set of all the state particles associated with this belief node, where
    * each particle is a pointer to a DiscreteState.
    *
    * Additionally, a belief node owns an ActionMapping, which stores the actions that have been
    * taken from this belief, as well as their associated statistics and subtrees.
    *
    * Key method is create_or_get_child()
    """
    def __init__(self, solver, id=None, parent_entry=None):
        if id is None:
            self.id = -1
        else:
            self.id = id

        self.solver = solver
        self.depth = -1
        self.action_map = None
        self.particle_pool = ParticlePool(self.solver.model.max_particle_count, self.solver.model.solver_type)   # The set of states that comprise the belief distribution of this belief node
        self.penalty_count = 0

        if parent_entry is not None:
            self.parent_entry = parent_entry
            # Correctly calculate the depth based on the parent node.
            self.depth = self.get_parent_belief().depth + 1
        else:
            self.parent_entry = None
            self.depth = 0


    def copy(self):
        bn = BeliefNode(self.solver, self.id, self.parent_entry)
        # share a reference to the action map
        bn.action_map = self.action_map
        bn.particle_pool = self.particle_pool
        return bn

    # Randomly select a History Entry
    def add_particle(self, particle, type):
        self.particle_pool.add_partcle(particle, type)

    # Randomly select a History Entry
    def sample_particle(self):
        return self.particle_pool.sample_particle()

    def sample_particle_of_POMCPOW(self):
        return self.particle_pool.sample_particle_of_POMCPOW()

    def set_create_current_particle(self, _bool):
        return self.particle_pool.set_create_current_particle(_bool)

    def get_num_total_particle(self):
        return self.particle_pool.get_num_total_particle()

    def get_create_current_particle(self):
        return self.particle_pool.get_create_current_particle()

    def get_diff_real_state_with_belief_state(self, gmuState, cellIndex, GRID_W):
        num, diff, distribution = self.particle_pool.get_numbers_of_state(gmuState, cellIndex, GRID_W)

        return num, diff, distribution
    # -------------------- Tree-related getters  ---------------------- #
    def get_parent_action_node(self):
        if self.parent_entry is not None:
            return self.parent_entry.map.owner
        else:
            return None

    def get_parent_belief(self):
        if self.parent_entry is not None:
            return self.parent_entry.map.owner.parent_entry.map.owner
        else:
            return None

    # Returns the last observation received before this belief
    def get_last_observation(self):
        if self.parent_entry is not None:
            self.parent_entry.get_observation()
        else:
            return None

    def get_last_action(self):
        if self.parent_entry is not None:
            return self.parent_entry.map.owner.parent_entry.get_action()
        else:
            return None

    def get_child(self, action, obs):
        node = self.action_map.get_action_node(action)
        if node is not None:
            return node.get_child(obs)
        else:
            return None

    def get_number_observation(self, action):
        actionNode = self.action_map.get_action_node(action)
        if actionNode is not None:
            return actionNode.get_count_child()
        else :
            return 0

    def get_visit_count_observation(self, action):
        actionNode = self.action_map.get_action_node(action)
        if actionNode is not None:
            return actionNode.get_visit_count()
        else:
            return 0
    def get_visit_count_specific_observation(self, action, obs):
        actionNode = self.action_map.get_action_node(action)
        if actionNode is not None:
            total_visit_count = actionNode.get_visit_count_obs()
            child_node = actionNode.get_child_entry(obs)
            if child_node is not None:
                    obs_visit_count = child_node.get_visit_count()

                    return obs_visit_count, total_visit_count
        else :
            return None

    def get_child_obs_entries(self, action):
        actionNode = self.action_map.get_action_node(action)

        if actionNode is not None:
            return actionNode.get_child_all_entries()

    def update_visit_count_observation(self, action, delta_n_visits):
        actionNode = self.action_map.get_action_node(action)

        if actionNode is not None:
            actionNode.update_visit_count(delta_n_visits)

    def update_visit_count_specific_observation(self, action, obs, delta_n_visits):
        actionNode = self.action_map.get_action_node(action)

        if actionNode is not None:
            child_node = actionNode.get_child_entry(obs)
            if child_node is not None:
                    child_node.update_visit_count(delta_n_visits)



    def child(self, action, obs):
        node = self.action_map.get_action_node(action)  # DiscreteActionMapping()
        if node is not None:
            child_node = node.get_child(obs)
            if child_node is None:
                return None, "NOOBS"
            return child_node, "OBS"
        else:
            return None, "NOACTION"

    # ----------- Core Methods -------------- #

    def create_or_get_child(self, action, obs):
        """
        Adds a child for the given action and observation, or returns a pre-existing one if it
        already existed.

        The belief node will also be added to the flattened node vector of the policy tree, as
        this is done by the BeliefNode constructor.
        :param action:
        :param obs:
        :return: belief node
        """
        action_node = self.action_map.get_action_node(action)
        if action_node is None:
            action_node = self.action_map.create_action_node(action)
            action_node.set_mapping(self.solver.observation_pool.create_observation_mapping(action_node))
        child_node, added = action_node.create_or_get_child(obs)

        if added:   # if the child node was added - it is new
            child_node.action_map = self.solver.action_pool.create_action_mapping(child_node)

        return child_node, added

    def create_child(self, action_node, obs):
        child_node, added = action_node.create_or_get_child(obs)

        if added:   # if the child node was added - it is new
            child_node.action_map = self.solver.action_pool.create_action_mapping(child_node)

        return child_node, added