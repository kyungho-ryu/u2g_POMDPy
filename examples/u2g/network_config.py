import math

class Config :
    # simulation space
    MAX_X = 2000
    MAX_Y = 2000 # meters
    GRID_W = 400 # grid width

    MAX_XGRID_N = int(MAX_X/GRID_W)
    MAX_YGRID_N = int(MAX_Y/GRID_W)
    MAX_GRID_INDEX = int(MAX_X/GRID_W) * int(MAX_Y/GRID_W) -1 #0~
    HEIGHT = 20
    GRID_CENTER = {}
    MAX_UAV_SPEED = 30 # [0, 30 m/s]
    UAV_RELOC_PERIOD = math.sqrt(MAX_X**2+MAX_Y**2)/MAX_UAV_SPEED #max seconds for diagonal range

    NUM_UAV = 25
    NUM_GMU = 10
    USER_DEMAND = 300 * pow(10, 3) # 300 Kbps

    WoE = 0.0005 # weight of energy consumption
    WoD = 0.0005 # weight of data rate