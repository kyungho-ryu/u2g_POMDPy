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
        return PositionAndGMUData(self.model, self.uav_position.copy(), self.all_rock_data, self.solver)

    def shallow_copy(self):
        """
        Creates a copy of this object's rock data to pass along to the new copy
        """
        new_gmu_data = self.copy_gmu_data(self.all_gmu_data)
        return PositionAndGMUData(self.model, self.uav_position.copy(), new_gmu_data, self.solver)

    def update(self, other_belief):
        self.all_rock_data = other_belief.data.all_rock_data

    def any_good_rocks(self):
        any_good_rocks = False
        for rock_data in self.all_rock_data:
            if rock_data.goodness_number > 0:
                any_good_rocks = True
        return any_good_rocks

    def create_child(self, rock_action, rock_observation):
        next_data = self.deep_copy()
        next_position, is_legal = self.model.make_next_position(self.grid_position.copy(), rock_action.bin_number)
        next_data.grid_position = next_position

        if rock_action.bin_number is ActionType.SAMPLE:
            rock_no = self.model.get_cell_type(self.grid_position)
            next_data.all_rock_data[rock_no].chance_good = 0.0
            next_data.all_rock_data[rock_no].check_count = 10
            next_data.all_rock_data[rock_no].goodness_number = -10

        elif rock_action.bin_number >= ActionType.CHECK:
            rock_no = rock_action.rock_no
            rock_pos = self.model.rock_positions[rock_no]

            dist = self.grid_position.euclidean_distance(rock_pos)
            probability_correct = self.model.get_sensor_correctness_probability(dist)
            probability_incorrect = 1 - probability_correct

            rock_data = next_data.all_rock_data[rock_no]
            rock_data.check_count += 1

            likelihood_good = rock_data.chance_good
            likelihood_bad = 1 - likelihood_good

            if rock_observation.is_good:
                rock_data.goodness_number += 1
                likelihood_good *= probability_correct
                likelihood_bad *= probability_incorrect
            else:
                rock_data.goodness_number -= 1
                likelihood_good *= probability_incorrect
                likelihood_bad *= probability_correct

            if np.abs(likelihood_good) < 0.01 and np.abs(likelihood_bad) < 0.01:
                # No idea whether good or bad. reset data
                # print "Had to reset RockData"
                rock_data = RockData()
            else:
                rock_data.chance_good = old_div(likelihood_good, (likelihood_good + likelihood_bad))

        return next_data

    def generate_legal_actions(self):
        return []

    def generate_smart_actions(self):
        pass



