from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
from pomdpy.pomdp import HistoricalData
import itertools


# Utility function
class GMUData(object):
    """
    Stores data about each rock
    """

    def __init__(self):
        self.t = 0

    def to_string(self):
        """
        Pretty printing
        """
        data_as_string = "t"
        return data_as_string

class PositionAndGMUData(HistoricalData):
    """
    A class to store the robot position associated with a given belief node, as well as
    explicitly calculated probabilities of goodness for each rock.
    """

    def __init__(self, model, grid_position, all_gmu_data, solver):
        self.model = model
        self.solver = solver
        self.uav_position = grid_position

        # List of RockData indexed by the rock number
        self.all_gmu_data = all_gmu_data
        # Holds reference to the function for generating legal actions
        if self.model.preferred_actions:
            self.legal_actions = self.generate_smart_actions
        else:
            self.legal_actions = self.generate_legal_actions

    @staticmethod
    def copy_gmu_data(other_data):
        new_rock_data = []
        [new_rock_data.append(GMUData()) for _ in other_data]
        for i, j in zip(other_data, new_rock_data):
            j.t = i.t
        return new_rock_data

    def copy(self):
        """
        Default behavior is to return a shallow copy
        """
        return self.shallow_copy()

    def deep_copy(self):
        """
        Passes along a reference to the rock data to the new copy of RockPositionHistory
        """
        return PositionAndGMUData(self.model, self.uav_position.copy(), self.all_gmu_data, self.solver)

    def shallow_copy(self):
        """
        Creates a copy of this object's rock data to pass along to the new copy
        """
        new_gmu_data = self.copy_gmu_data(self.all_gmu_data)
        return PositionAndGMUData(self.model, self.uav_position.copy(), new_gmu_data, self.solver)

    def update(self, other_belief):
        self.uav_position = other_belief.data.uav_position


    def create_child(self, u2g_action, u2g_observation):
        next_data = self.deep_copy()
        next_data.grid_position = u2g_action.UAV_deployment

        return next_data

    def generate_legal_actions(self):
        return []

    def generate_smart_actions(self):
        pass



