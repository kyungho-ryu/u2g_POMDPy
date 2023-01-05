import os
import pandas as pd
class Trajectory() :
    def __init__(self, traj):
        self.traj = traj
        self.updated_time = 0
        self.trajLen = len(traj["x"])
    def get_trajectory(self, i):
        try :
            traj = self.traj["x"][i], self.traj["y"][i]
        except :
            k = i - self.trajLen
            traj = self.traj["x"][k], self.traj["y"][k]
        return traj

    def update_time(self, t):
        if t >= self.trajLen :
            self.updated_time = t-self.trajLen
        else :
            self.updated_time = t

_list = os.listdir('/home/kyungho/project/U2G_POMDPy/mobility/trajectory/')

NumOfMO = 25
traj = {}
for i in range(NumOfMO) :
    file = "/home/kyungho/project/U2G_POMDPy/mobility/trajectory/" + str(_list[i])

    traj[i] = Trajectory(pd.read_csv(file))

print(traj[0].trajLen)
print(traj[0].get_trajectory(2569))
traj[0].update_time(2570)
print(traj[0].updated_time)