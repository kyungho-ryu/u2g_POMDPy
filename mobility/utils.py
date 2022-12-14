from collections import OrderedDict
import copy, random
class FixSizeOrderedDict(OrderedDict) :
    def __init__(self, *args,  max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max >0 :
            if len(self) > self._max :
                self.popitem(False)

def set_coordinate(x, y, x0, y0) :
    x, y = (x - x0), (y - y0)

    if x <0 or y<0 :
        return -1
    else :
        return x, y

def get_cellCoordinate(x, y, diameterofCell) :
    x, y= int(x //diameterofCell), int(y //diameterofCell)

    if x <0 or y<0 :
        print("asdasd")
        exit()
        return -1
    else :
        return x, y


def get_condition_prob_in_likelihood(previousRO, currentRO) :
    RO = []
    for i in range(len(currentRO)) :
        RO.append((currentRO[i][0], currentRO[i][1] -1))

    union = len(set(previousRO + RO))
    intersection = len(previousRO + RO) - union

    state_transition_prob = intersection / union

    return union, intersection, state_transition_prob

def get_id_of_gmu(ro) :
    return int(ro[2:])

def create_random_position_in_cell(x,y, cellWidth) :
    x = random.randint(cellWidth*x+1, cellWidth*(x+1)-1)
    y = random.randint(cellWidth*y+1, cellWidth*(y+1)-1)

    return x, y

def getGridIndex(_x, _y, MAX_XGRID_N):
    return _y * MAX_XGRID_N + _x

def add_noise_to_trajectory(trajectory) :
    length = 2* len(trajectory) - 3
    noise_index = random.randint(0, length)

    x_index = int(noise_index//2)
    y_index = int(noise_index%2)

    oldX, oldY = trajectory[x_index]
    if y_index == 0 :
        trajectory[x_index] = oldX +1, oldY
    else :
        trajectory[x_index] = oldX, oldY +1

    return trajectory
