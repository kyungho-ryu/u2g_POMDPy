import pandas as pd
import numpy as np
import pathlib
import csv

def get_cellCoordinate(index, MAX_XGRID_N=5) :
    x = index%MAX_XGRID_N
    y = index//MAX_XGRID_N

    return x+0.5, y+0.5

deployment = {"x" : [], "y" : []}

T = 2074

file = str(pathlib.Path().resolve()) + "/csvFile/UAV_deployment_original.csv"

f = open(file, 'r')
cr = csv.reader(f)
i = 0
for line in cr :
    if i == T :
        _temp = line
        break
    i +=1
print(_temp)
WorkUavs = []
for index in _temp :
    if index not in WorkUavs :
        x, y = get_cellCoordinate(int(index))
        deployment["x"].append(x)
        deployment["y"].append(y)
        WorkUavs.append(index)

deployment = pd.DataFrame(deployment)
fileName = str(pathlib.Path().resolve()) + "/csvFile/UAV_deployment.csv"

deployment.to_csv(fileName,
            sep=',',
            na_rep='NaN',
            index = False)
