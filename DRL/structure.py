from enum import IntEnum, unique
@unique
class DRLType(IntEnum) :
    IS_A2CModel=0
    OS_A2CModel=1
    IS_PPOModel=2
    OS_PPOModel=3


