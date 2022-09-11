import pandas as pd
from utils import set_coordinate, get_cellCoordinate, getGridIndex
import numpy as np

# traj = {i : [] for i in range(10)}
#
# for i in range(10) :
#     file = "/home/kyungho/project/U2G_POMDPy/mobility/original_trajectory/MO" + str(i) +"_traj.csv"
#     temp = pd.read_csv(file)
#
#     temp = temp.to_dict()
#     for j in range(int(len(temp["x"])-2), -1, -1) :
#         temp["x"][len(temp["x"])] = (temp["x"][j])
#         temp["y"][len(temp["y"])] = (temp["y"][j])
#         temp["speed"][len(temp["speed"])] = (temp["speed"][j])
#         temp["pos"][len(temp["pos"])] = (temp["pos"][j])
#         temp["angle"][len(temp["angle"])] = (temp["angle"][j])
#
#     for k in range(4) :
#         for j in range(1, int(len(temp["x"])), 1) :
#             temp["x"][len(temp["x"])] = (temp["x"][j])
#             temp["y"][len(temp["y"])] = (temp["y"][j])
#             temp["speed"][len(temp["speed"])] = (temp["speed"][j])
#             temp["pos"][len(temp["pos"])] = (temp["pos"][j])
#             temp["angle"][len(temp["angle"])] = (temp["angle"][j])
#
#     traj = pd.DataFrame(temp)
#     traj.to_csv('/home/kyungho/project/U2G_POMDPy/mobility/trajectory/MO' + str(i) +"_traj.csv",
#                 sep=',',
#                 na_rep='NaN',
#                 index = False)

traj = {i : [] for i in range(10, 40)}

for i in range(10, 40) :
    file = "/home/kyungho/project/U2G_POMDPy/mobility/original_trajectory/MO" + str(i) +"_traj.csv"
    temp = pd.read_csv(file)

    temp = temp.to_dict()
    for j in range(int(len(temp["x"])-2), -1, -1) :
        temp["x"][len(temp["x"])] = (temp["x"][j])
        temp["y"][len(temp["y"])] = (temp["y"][j])
        temp["speed"][len(temp["speed"])] = (temp["speed"][j])
        temp["pos"][len(temp["pos"])] = (temp["pos"][j])
        temp["angle"][len(temp["angle"])] = (temp["angle"][j])


    traj = pd.DataFrame(temp)
    traj.to_csv('/home/kyungho/project/U2G_POMDPy/mobility/trajectory/MO' + str(i) +"_traj.csv",
                sep=',',
                na_rep='NaN',
                index = False)