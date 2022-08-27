from enum import IntEnum, unique
@unique
class ActionType(IntEnum) :
    Random=0
    Near=1
    NN=2

