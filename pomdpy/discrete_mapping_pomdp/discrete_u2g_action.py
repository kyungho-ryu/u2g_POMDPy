import abc
from pomdpy.pomdp.point import Point


class DiscreteU2GAction(Point):

    def __init__(self, deployment):
        self.UAV_deployment = deployment

    def __hash__(self):
        return self.UAV_deployment

    def __eq__(self, other_discrete_action):
        return self.UAV_deployment == other_discrete_action.UAV_deployment

    @abc.abstractmethod
    def print_action(self):
        """
        Pretty prints the action type
        :return:
        """

    @abc.abstractmethod
    def to_string(self):
        """
        Returns a String version of the action type
        :return:
        """
    @abc.abstractmethod
    def copy(self):
        """
        Returns a proper copy of the Discrete Action
        :return:
        """
