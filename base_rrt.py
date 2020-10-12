import random
import numpy as np
import time
from abc import ABC, abstractmethod


class Node(object):
    """
    RRT Node
    """
    def __init__(self, state):
        super().__init__()
        self.state = state.copy()
        self.path = []
        self.parent = None
        self.cost = 0.0


class BaseRRT(ABC, object):
    """
    Base class for RRT planning. There are four major components for RRT:
    Methods:
        * sample: sample the state space, it can be uniform or biased.
        * choose_parent: choose the most promising sample, 
        i.e. with the minimum cost, to extend.
        * steer: steer a node to a goal region.
        * valid_state_check: collision checking. But in our design, collision checking is embedded in the env, so we ignore it in the RRT class.
    """
    def __init__(self, env, **kwargs):
        super().__init__()
        self.env = env
        bounds = env.get_bounds()
        self.state_bounds = bounds['state_bounds']
        self.cbounds = bounds['cbounds']

        self.node_list = []
        self.start = None
        self.goal = None

        self.planning_time = 0.0
        self.path = None

    def set_start_and_goal(self, start, goal):
        """ Set the start and goal states for planning
        Args:
            start (np.ndarray)
            goal (np.ndarray) 
        """
        assert len(start) == len(goal) == len(self.state_bounds), \
            'dimensions of the start or the goal are wrong'

        assert self.env.valid_state_check(start) and \
            self.env.valid_state_check(goal), \
            'start or goal states are not valid'

        self.start = Node(start)
        self.goal = Node(goal)

    @abstractmethod
    def sample(self, goal):
        """Sample the state space. 
        """
        pass

    @abstractmethod
    def choose_parent(self, rnd_node):
        """Choose the most promising node to extend
        """
        pass

    @abstractmethod
    def steer(self, from_node, to_node):
        """Steer from_node to to_node.
        Args:
            from_node (Node): a node in the tree
            to_node (Node): a sample node
        """
        pass

    def generate_final_course(self, node):
        """Use to generate a course from the start node
        """
        path = []
        while node.parent is not None:
            path = node.path[1:] + path
            node = node.parent
        path = [node.state] + path
        return path

    @staticmethod
    def near(node_list, rnd_node, metric, delta):
        dis_list = np.array([metric(node, rnd_node) for node in node_list])
        ind = np.flatnonzero(dis_list <= delta)
        if len(ind) == 0:
            return None
        else:
            return [node_list[i] for i in ind]

    @staticmethod
    def nearest(node_list, rnd_node, metric):
        dis_list = np.array([metric(node, rnd_node)
                             for node in node_list])
        min_ind = np.argmin(dis_list)
        return node_list[min_ind], dis_list[min_ind]




