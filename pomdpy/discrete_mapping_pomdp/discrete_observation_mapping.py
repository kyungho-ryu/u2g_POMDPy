from pomdpy.pomdp import ObservationMapping, ObservationMappingEntry, BeliefMappingNode
import hashlib

class DiscreteObservationMap(ObservationMapping):
    """
    A concrete class implementing ObservationMapping for a discrete set of observations.
    *
    * The mapping entries are stored in a dictionary, which maps each observation
    * to its associated entry in the mapping
    * The dictionary returns None if an observation is not yet stored in the dictionary
    """
    def __init__(self, action_node, agent):
        super(DiscreteObservationMap, self).__init__(action_node)
        self.agent = agent
        self.child_map = {}
        self.total_visit_count = 0

    def get_belief(self, disc_observation):
        key = self.get_key(disc_observation.observed_gmu_status)
        entry = self.get_entry(key)
        if entry is None:
            return None
        else:
            return entry.child_node

    def create_belief(self, disc_observation, belief_map):
        entry = DiscreteObservationMapEntry()
        entry.map = self
        entry.observation = disc_observation
        entry.child_node = BeliefMappingNode(self.agent, belief_map)
        key = self.get_key(disc_observation.observed_gmu_status)
        self.child_map.__setitem__(key, entry)
        return entry.child_node

    def update_child_node(self, disc_observation, child_node):
        key = self.get_key(disc_observation.observed_gmu_status)
        self.child_map[key].child_node = child_node


    def delete_child(self, obs_mapping_entry):
        self.total_visit_count -= obs_mapping_entry.visit_count
        del self.child_map[obs_mapping_entry.observation]

    def get_child_entries(self):
        return_entries = []
        for key in list(self.child_map.keys()):
            return_entries.append(self.child_map.get(key))
        return return_entries

    def get_child_nodes(self):
        return_nodes = []
        for key in list(self.child_map.keys()):
            return_nodes.append(self.child_map.get(key).child_node)
        return return_nodes

    def get_number_child_entries(self):
        return len(self.child_map)

    def get_entry(self, obs):
        for i in list(self.child_map.values()):
            if obs == i.observation:
                return i
        return None

    def get_key(self, obs):
        return hashlib.sha256(str(obs).encode()).hexdigest()

    def get_visit_count(self):
        return self.total_visit_count

class DiscreteObservationMapEntry(ObservationMappingEntry):
    """
    A concrete class implementing ObservationMappingEntry for a discrete set of observations.
    *
    * Each entry stores a pointer back to its parent map, the actual observation it is associated with,
    * and its child node and visit count.
    """

    def __init__(self):
        self.map = None  # DiscreteObservationMap
        self.observation = None     # DiscreteObservation
        # The child node of this entry (should always be non-null).
        self.child_node = None  # belief node
        self.visit_count = 0

    def get_observation(self):
        return self.observation.copy()

    def get_visit_count(self):
        return self.visit_count

    def update_visit_count(self, delta_n_visits):
        self.visit_count += delta_n_visits
        self.map.total_visit_count += delta_n_visits
