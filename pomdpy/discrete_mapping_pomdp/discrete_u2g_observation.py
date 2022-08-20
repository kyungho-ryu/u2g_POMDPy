import abc
from pomdpy.pomdp import Point


class DiscreteU2GObservation(Point):

    def __init__(self, observed_gmu_status):
        self.observed_gmu_status = observed_gmu_status

    def __hash__(self):
        return self.observed_gmu_status

    def __eq__(self, other_discrete_observation):
        return self.observed_gmu_status == other_discrete_observation.observed_gmu_status

    @abc.abstractmethod
    def copy(self):
        """
        :return:
        """
    @abc.abstractmethod
    def print_observation(self):
        """
        pretty printing
        :return:
        """

    @abc.abstractmethod
    def to_string(self):
        """
        Returns a String version of the observation
        :return:
        """