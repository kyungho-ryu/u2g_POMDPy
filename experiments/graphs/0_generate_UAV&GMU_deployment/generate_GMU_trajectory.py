import pandas as pd
import numpy as np
import pathlib

def set_coordinate(x, y, x0=477, y0=634) :
    x, y = (x - x0), (y - y0)

    if x <0 or y<0 :
        return -1
    else :
        return x, y

def get_cellCoordinate(x, y, diameterofCell=400) :
    x, y= int(x //diameterofCell), int(y //diameterofCell)

    if x <0 or y<0 :
        return -1
    else :
        return x, y


GMU_ids = [i for i in range(20)]
deployment = {"x" : [], "y" : [], "count" : []}
for i in range(5) :
    for j in range(5) :
        deployment["x"].append(i)
        deployment["y"].append(j)
        deployment["count"].append(0)

T = 2074
T = 4 * (T +2)

for i in GMU_ids :
    file = "/home/kyungho/project/U2G_POMDPy/mobility/trajectory/MO" + str(i) +"_traj.csv"
    temp = pd.read_csv(file)

    # traj["id"].append(i)

    x, y = set_coordinate(temp["x"][T], temp["y"][T])
    x, y = get_cellCoordinate(x, y)
    index = 5*x + y
    deployment["count"][index] +=1

deployment = pd.DataFrame(deployment)
fileName = str(pathlib.Path().resolve()) + "/csvFile/GMU_deployment.csv"

deployment.to_csv(fileName,
            sep=',',
            na_rep='NaN',
            index = False)
