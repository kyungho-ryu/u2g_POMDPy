import pandas as pd
from utils import set_coordinate, get_cellCoordinate, getGridIndex
import numpy as np

traj = {i : [] for i in range(10)}
MinLen = np.inf
for i in range(10) :
    file = "/home/kyungho/project/U2G_POMDPy/mobility/trajectory/MO" + str(i) +"_traj.csv"
    temp = pd.read_csv(file)
    for j in range(len(temp["x"])) :
        _coordinate = set_coordinate(temp["x"][j], temp["y"][j], 477, 634)
        coordinate = get_cellCoordinate(_coordinate[0], _coordinate[1], 400)
        cell = getGridIndex(coordinate[0], coordinate[1], 5)
        traj[i].append(cell)

    if len(traj[i]) < MinLen :
        MinLen = len(traj[i])

for i in range(10) :
    traj[i] = traj[i][0:MinLen]

traj = pd.DataFrame(traj)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,15))
sns.heatmap(data = traj.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()