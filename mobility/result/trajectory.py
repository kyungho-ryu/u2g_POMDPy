import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

id1 = 50
id2 = 49
file1 = "/home/kyungho/project/POMDPy/mobility/trajectory/MO" + str(id1) + "_traj.csv"
file2 = "/home/kyungho/project/POMDPy/mobility/trajectory/MO" + str(id2) + "_traj.csv"
traj1 = pd.read_csv(file1)
traj2 = pd.read_csv(file2)

NUM = 10
interval = 4
index = 0



for j in range(10) :
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    colors = cm.rainbow(np.linspace(0, 1, NUM))
    for i in range(index, index+(NUM*interval), interval) :
        x1.append(traj1["x"][i])
        x2.append(traj2["x"][i])
        y1.append(traj1["y"][i])
        y2.append(traj2["y"][i])

        index +=interval

    plt.plot(x1, y1)
    plt.plot(x2, y2, '--')
    plt.show()

