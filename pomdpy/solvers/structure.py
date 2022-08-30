from enum import IntEnum, unique
@unique
class SolverType(IntEnum) :
    POMCP_DPW=0
    POMCP_POW=1
    POMCP_DPW_WITH_NN=2
    POMCP_POW_WITH_NN=3

