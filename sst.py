import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
from os.path import abspath, dirname, join
import sys
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import control as oc
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    sys.path.insert(
        0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import control as oc
from functools import partial

import xml.etree.ElementTree as ET
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../DynamicalEnvs/")
from dubin import DubinEnv

# xml namespace, move it from graphml
xmlns = ' xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"'

EXACT_SOLUTION = 'Exact solution'
APPROXIMATE_SOLUTION = 'Approximate solution'


class SST(object):
    def __init__(self, robot_env: DubinEnv):
        self.robot_env = robot_env
        bounds = self.robot_env.get_bounds()
        self.state_bounds = bounds['state_bounds']
        self.control_bounds = bounds['cbounds']

        # add the state space
        space = ob.RealVectorStateSpace(len(self.state_bounds))
        bounds = ob.RealVectorBounds(len(self.state_bounds))
        # add state bounds
        for k, v in enumerate(self.state_bounds):
            bounds.setLow(k, float(v[0]))
            bounds.setHigh(k, float(v[1]))
        space.setBounds(bounds)
        self.space = space
        # add the control space
        cspace = oc.RealVectorControlSpace(space, len(self.control_bounds))
        # set the bounds for the control space
        cbounds = ob.RealVectorBounds(len(self.control_bounds))
        for k, v in enumerate(self.control_bounds):
            cbounds.setLow(k, float(v[0]))
            cbounds.setHigh(k, float(v[1]))
        cspace.setBounds(cbounds)
        self.cspace = cspace
        # define a simple setup class
        self.ss = oc.SimpleSetup(cspace)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(
            partial(self.isStateValid, self.ss.getSpaceInformation())))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        # set the planner
        si = self.ss.getSpaceInformation()
        planner = oc.SST(si)
        self.ss.setPlanner(planner)
        si.setPropagationStepSize(self.robot_env.step_dt)

        self.start = None
        self.goal = None
        self.planning_time = 0.0
        self.path = None

    def obRealVector2array(self, realvector):
        n = len(self.state_bounds)
        array = np.zeros(n)
        for i in range(n):
            array[i] = realvector[i]
        return array

    def array2obRealVector(self, array, realvector):
        n = len(self.state_bounds)
        for i in range(n):
            realvector[i] = array[i]
        return None

    def isStateValid(self, spaceInformation, state):
        state_array = self.obRealVector2array(state)
        return self.robot_env.valid_state_check(state_array) \
            and spaceInformation.satisfiesBounds(state)

    def propagate(self, start, control, duration, state):
        state_array = self.obRealVector2array(start)
        control_array = self.obRealVector2array(control)
        state_array = self.robot_env.motion(
            state_array, control_array, duration)
        for i in range(len(state_array)):
            state[i] = state_array[i]

    def set_start_and_goal(self, start: np.ndarray, goal: np.ndarray):
        self.start = ob.State(self.space)
        self.goal = ob.State(self.space)
        self.array2obRealVector(start, self.start)
        self.array2obRealVector(goal, self.goal)

    def check_reach(self, path):
        goal_array = self.obRealVector2array(self.goal)
        if self.robot_env.reach(path[-1, :len(self.state_bounds)], goal_array):
            return True
        else:
            return False

    def planning(self, runtime=50.0):
        if self.start is None and self.goal is None:
            return False
        self.ss.clear()

        dt = 5.0
        tic = time.time()
        find_exact_solution = False
        self.ss.setStartAndGoalStates(self.start, self.goal, 1.5)
        path_length = 10000.0
        for _ in np.arange(0.0, runtime+dt, dt):
            print(f"Total run time {time.time()-tic}")
            self.ss.solve(dt)
            path = self.ss.getSolutionPath()
            if self.ss.haveExactSolutionPath():
                if path_length - path.length() > 1.0:
                    print(
                        f'path quality improve {path_length - path.length()} seconds')
                    path_length = path.length()
                else:
                    print(
                        f'path quality only improve {path_length - path.length()} seconds')
                    break
        path.interpolate()
        path_matrix = path.printAsMatrix()
        path = np.array([j.split()
                         for j in path_matrix.splitlines()], dtype=float)
        if self.check_reach(path):
            find_exact_solution = True
        toc = time.time()
        self.planning_time = toc - tic
        self.path = path

        # get planner data
        pd = ob.PlannerData(self.ss.getSpaceInformation())
        self.ss.getPlannerData(pd)
        pd.computeEdgeWeights()
        xmlstring = pd.printGraphML()
        self.planner_data = self.xml2tree(xmlstring)
        self.reach_exactly = find_exact_solution
        return find_exact_solution

    @staticmethod
    def xml2tree(xmlstring):
        root = ET.fromstring(xmlstring.replace(xmlns, ''))
        nodes = np.array([node[0].text.split(',')[:2]
                          for node in root.iter('node')], dtype=float)
        edges_weight = np.array([edge[0].text
                                 for edge in root.iter('edge')], dtype=float)
        edges = np.array([[edge.attrib['source'].replace('n', ''),
                           edge.attrib['target'].replace('n', '')]
                          for edge in root.iter('edge')], dtype=int)
        return {'nodes': nodes, 'edges': edges, 'edges_weight': edges_weight}

    def get_planner_data(self):
        return self.planner_data.copy()

    def get_path(self):
        return self.path.copy()

    def get_path_len(self):
        if self.path is None:
            return None
        return np.sum(self.path[:, -1])


def extract_full_data(env, path):
    local_maps = []
    state_len = len(env.state_bounds)
    control_len = len(env.cbounds)
    for state in path:
        env.state_ = state[:state_len]
        local_maps.append(env._obs()['local_map'])
    return {'states': path[:, :state_len], 'controls': path[1:, state_len:state_len+control_len], 'local_maps': np.array(local_maps)}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_num', type=int, default=1000)
    parser.add_argument('--low', type=float, default=5.0)
    parser.add_argument('--high', type=float, default=15.0)
    parser.add_argument('--logdir', type=str, default='log')
    return parser.parse_known_args()[0]

def collect(args = get_args()):
    env = DubinEnv()
    save_dir = args.logdir + '/dubin_path/env1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(args.traj_num):
        env.reset(low=5.0, high=20)
        start, goal = env.state, env.goal
        planner = SST(env)
        planner.set_start_and_goal(start, goal)
        planner.planning(40)
        data = extract_full_data(env, path=planner.get_path())
        pickle.dump(data, open(f'{save_dir}/path{i}.pkl', 'wb'))
        print(f'save traj {i}')

def test():
    env = DubinEnv()
    env.reset(low=5.0, high=20)
    start, goal = env.state, env.goal
    planner = SST(env)
    planner.set_start_and_goal(start, goal)
    planner.planning(40)
    data = extract_full_data(env, path=planner.get_path())
    pickle.dump({'start': start, 'goal': goal}, open('start_goal.pkl', 'wb'))
    pickle.dump(data, open('path.pkl', 'wb'))
if __name__ == "__main__":
    collect()

