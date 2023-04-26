from .utils import FixSizeOrderedDict,get_id_of_gmu
from .mobility_config import MConfig


# Trajectory Grid
class TG () :
    def __init__(self, logger):
        self.leafCells = {}
        self.logger = logger

    def add_new_trajectory(self, new_loc, current_loc, id, previous_t):
        if current_loc != [] :
            self.leafCells[current_loc].update_traHash(id, new_loc, previous_t)
        if not new_loc in self.leafCells:
            self.leafCells[new_loc] = Cell()



        # update the density
        self.leafCells[new_loc].visit()

    # find reference objects
    def lookup(self, id, trajectories, MOS, NUMMOS):

        def find_candidate_objs (id, traj, candidate, MOS, NUMMOS, last_index):   # candiate = id, t
            candidate_objs = []
            if traj not in self.leafCells :
                return candidate_objs

            self.logger.debug("current selected {}:".format(candidate))
            self.logger.debug("candidate : {}".format(list(self.leafCells[traj].trajectories.keys())))

            for k, v in self.leafCells[traj].trajectories.items() :
                candidate_id = get_id_of_gmu(k[0])
                # if k[0] != id and candidate_id >= NUMMOS:
                if k[0] != id and candidate_id >= NUMMOS:
                    if candidate ==[] or k in candidate :
                        if last_index :
                            candidateOBJ = k

                            if MOS[candidate_id].get_current_time() <= int(k[1]) + MConfig.Min_remaining_trajectory:
                                continue
                        else :
                            candidateOBJ = k[0], k[1]+1

                        candidate_objs.append(candidateOBJ)

            return candidate_objs

        referenceOBJs = []

        self.logger.debug("{}'s backward trajesctories :{}".format(id, trajectories))
        for i in range(len(trajectories)) :
            referenceOBJs = (find_candidate_objs(id, trajectories[i], referenceOBJs, MOS, NUMMOS, i==len(trajectories)-1))
            if referenceOBJs == [] : return referenceOBJs
        return referenceOBJs

    def get_densityCell(self, traj):
        return self.leafCells[traj].get_density()
# leaf node in TG
# key = (x,y)
class Cell () :
    def __init__(self):
        self.density = 0        # provides prior information for Prediction Filter (E.q (14))
        self.density_transtion = {}
        self.trajectories = FixSizeOrderedDict(max=MConfig.H)

    def update_traHash(self, id, new_loc, previous_t):
        key = (id, previous_t)
        self.trajectories[key] = TraHash(new_loc[0], new_loc[1])

        if new_loc not in self.density_transtion :
            self.density_transtion[new_loc] = 0
        self.density_transtion[new_loc] +=1

    def visit(self):
        self.density +=1

    def get_density(self):
        return self.density * MConfig.C_density

    def get_transition_density(self, next_loc):
        return self.density_transtion[next_loc] * MConfig.C_density

# hash table to store the trajectories passing the cell
class TraHash :
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_loc(self):
        return self.x, self.y