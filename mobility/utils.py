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
        return -1
    else :
        return x, y


def get_cellIndex(x, y, scale) :
    return x+scale*y

def get_state_transition_prob(previousRO, currentRO) :
    RO = []
    for i in range(len(currentRO)) :
        RO.append((currentRO[i][0], currentRO[i][1] -1))

    union = len(set(previousRO + RO))
    intersection = len(previousRO + RO) - union

    state_transition_prob = intersection / union

    return union, intersection, state_transition_prob


def create_random_position_in_cell(x,y, cellWidth) :
    x = random.randint(cellWidth*x, cellWidth*(x+1))
    y = random.randint(cellWidth*y, cellWidth*(y+1))

    return x, y

def getGridIndex(_x, _y, MAX_XGRID_N):
    return _y * MAX_XGRID_N + _x
