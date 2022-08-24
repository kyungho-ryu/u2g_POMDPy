from __future__ import print_function


class U2GReward():

    def __init__(self, totalDnRate, totalA2GEnergy, totalA2AEnergy, totalPropEnergy, NumActiveUav):
        self.totalDnRate = totalDnRate
        self.totalA2GEnergy = totalA2GEnergy
        self.totalA2AEnergy = totalA2AEnergy
        self.totalPropEnergy = totalPropEnergy
        self.NumActiveUav = NumActiveUav

    def get_reward(self) :
        return self.totalA2GEnergy, self.totalA2AEnergy, self.totalPropEnergy, self.totalDnRate

    def get_NumActiveUav(self):
        return self.NumActiveUav