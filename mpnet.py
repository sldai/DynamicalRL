
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from net import Actor, Conv, SamplerImage, SamplerPcs
import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../DynamicalEnvs/")
from car import CarEnv
import torch
import pickle

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

def get_normal_state(env):
    state = env.state_
    goal = env.goal_
    state[0] /= 20
    state[1] /= 20
    state[2] /= np.pi
    state[3] /= env.state_bounds[3,1]   
    goal[0] /= 20
    goal[1] /= 20
    goal[2] /= np.pi
    goal[3] /= env.state_bounds[3,1] 
    norm_state = np.array([state[0]-goal[0], state[1]-goal[1], state[2], state[3], goal[2]])
    return norm_state

def recover_state(state_):
    state = np.zeros(4)
    state[:3] = state_[:3]
    state[0] *= 20
    state[1] *= 20
    state[2] *= np.pi
    return state

class MPNet(object):
    """
    RRT using time to reach as the cost. It always tries to extend the node with the lowest heuristic to the sample.
    """
    def __init__(self, env, policy, sampler, goal_sample_rate=5, max_iter=500, verbose = False, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.env = env
        bounds = env.get_bounds()
        self.state_bounds = bounds['state_bounds']
        self.cbounds = bounds['cbounds']
        self.dt = self.env.step_dt
        self.node_list = []

        self.policy = policy
        self.sampler = sampler

        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.epsilon = 10.0 # near radius
        
        self.t_propagate = 1.0
        self.t_max_extend = self.t_propagate

        self.start = None
        self.goal = None

        self.verbose = verbose
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

    def planning(self):
        """
        rrt path planning
        """
        if self.start is None or self.goal is None:
            raise ValueError('start or goal is not set')
        tic = time.time()
        self.node_list = [self.start]
        path = None
        reach_exactly = False
        best_node = None

        x_rnd = self.start.state.copy()
        x_goal = self.goal.state.copy()
        for i in range(self.max_iter):
            if self.verbose:
                print('Iteration {}: {} nodes'.format(i, len(self.node_list)))

            good_sample = False
            while not good_sample:
                x_rnd_node = self.node_list[-1]
                x_rnd = self.node_list[-1].state.copy()
                self.env.state_ = x_rnd
                self.env.goal_ = x_goal
                s = get_normal_state(self.env)
                Z = self.env._obs()['local_map']
                s = np.array([s])
                Z = np.array([Z])

                tmp_samples = self.sampler(s, Z)[0].detach().cpu().numpy()
                state_next = recover_state(tmp_samples[0])
                if self.env.valid_state_check(state_next):
                    rnd_node = Node(state_next)
                    good_sample = True

            parent_node = x_rnd_node
            new_node_list = self.steer(parent_node, rnd_node, 2*self.t_propagate, self.t_propagate)
            # parent_node = x_rnd_node
            # rnd_node.parent = parent_node
            # rnd_node.path = [rnd_node.state]
            # new_node_list = [rnd_node]
            # self.node_list += new_node_list

            if len(new_node_list)>0:
                if self.se2_metric(new_node_list[-1], self.goal)<5 and not self.env.reach(new_node_list[-1].state, self.goal.state):
                    new_node_list += self.steer(new_node_list[-1], self.goal, 2*self.t_propagate, self.t_propagate)
                if self.env.reach(new_node_list[-1].state, self.goal.state):
                    near_node = new_node_list[-1]
                    reach_exactly = True
                    if best_node == None or near_node.cost<best_node.cost:
                        best_node = near_node
                    if self.verbose:
                        print('Time: {}'.format(time.time()-tic))
                        print('Path length: {}'.format(best_node.cost))
                    break
        if best_node == None:
            best_node, dis = self.nearest(self.node_list, self.goal, self.se2_metric)
        path = self.generate_final_course(best_node)
        toc = time.time()
        self.planning_time = toc-tic
        self.path = path
        self.reach_exactly = reach_exactly
        return path

    def sample(self, goal):
        """Uniform sample the state space, 
        it's possible to directly sample the goal state.
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            state = np.random.uniform(
                self.state_bounds[:, 0], self.state_bounds[:, 1])
            rnd = Node(state)

        else:  # goal point sampling
            rnd = Node(self.goal.state)
        return rnd
    
    @staticmethod
    def Eu_metric(src_node, dst_node):
        return np.linalg.norm(dst_node.state[:2]-src_node.state[:2])

    @staticmethod
    def se2_metric(src_node, dst_node):
        src_x = np.zeros(4)
        src_x[:2] = src_node.state[:2]
        src_x[2], src_x[3] = np.cos(src_node.state[2]), np.sin(src_node.state[2])

        dst_x = np.zeros(4)
        dst_x[:2] = dst_node.state[:2]
        dst_x[2], dst_x[3] = np.cos(dst_node.state[2]), np.sin(dst_node.state[2])
        return np.linalg.norm(dst_x - src_x)

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

    def choose_parent(self, rnd_node):
        """The chosen node has the lowest heuristic.
        """
        parent_node, min_c_cost = self.nearest(self.node_list, rnd_node, self.se2_metric)
        return parent_node

    def steer(self, from_node, to_node, t_max_extend=10.0, t_tree=5.0):
        """Use RL policy to steer from_node to to_node
        """
        parent_node = from_node
        new_node = Node(from_node.state)
        new_node.parent = parent_node
        new_node.path.append(new_node.state)
        state = new_node.state
        goal = to_node.state.copy()

        env = self.env
        env.state_ = state
        env.goal_ = goal

        dt = env.step_dt
        n_max_extend = round(t_max_extend/dt)
        n_tree = round(t_tree/dt)
        new_node_list = []
        n_extend = 0
        while not (n_extend>n_max_extend or env.reach(new_node.state, goal) or not env.valid_state_check(new_node.state)):
            obs = env._obs()['dynamics']
            action, _ = self.policy(obs.reshape((1,-1)))
            action = action.detach().cpu().numpy()[0]
            obs, rewards, done, info = env.step(action)

            new_node.state = env.state_
            new_node.path.append(new_node.state)
            n_extend += 1
            if (not info['collision']) and (n_extend % n_tree == 0 or info['goal']):
                new_node_list.append(new_node)
                parent_node = new_node
                new_node = Node(parent_node.state)
                new_node.parent = parent_node
                new_node.path.append(new_node.state)
                
        self.node_list += new_node_list
        self.propagate_cost_to_leaves(from_node)
        return new_node_list

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost+self.dt*(len(node.path)-1) 
                self.propagate_cost_to_leaves(node)

    def generate_final_course(self, node):
        """Use to generate a course from the start node
        """
        path = []
        while node.parent is not None:
            path = node.path[1:] + path
            node = node.parent
        path = [node.state] + path
        return path

def sample_pcs(env, num):
    """Sample #num point clouds in a given env
    """
    bounds = env.get_bounds()
    workspace_bounds = bounds['workspace_bounds']
    low = workspace_bounds[:, 0]
    high = workspace_bounds[:, 1]

    pcs = np.zeros((0, len(workspace_bounds)))
    while True:
        samples = np.random.uniform(low=low, high=high, size=(num, 2))
        ind = np.flatnonzero(env.valid_point_check(samples) == 0)
        pcs = np.concatenate((pcs, samples[ind]), axis=0)
        if len(pcs) >= num:
            pcs = pcs[:num]
            break
    return pcs

class MPNetPcs(MPNet):
    def planning(self):
        """
        rrt path planning
        """
        if self.start is None or self.goal is None:
            raise ValueError('start or goal is not set')
        tic = time.time()
        self.node_list = [self.start]
        path = None
        reach_exactly = False
        best_node = None

        x_rnd = self.start.state.copy()
        x_goal = self.goal.state.copy()
        num_pcs = 4000
        Z = sample_pcs(self.env, num_pcs)
        Z[:,0] /= 20
        Z[:,1] /= 20
        Z = Z.reshape((num_pcs*2,))
        for i in range(self.max_iter):
            if self.verbose:
                print('Iteration {}: {} nodes'.format(i, len(self.node_list)))

            good_sample = False
            while not good_sample:
                x_rnd_node = self.node_list[-1]
                x_rnd = self.node_list[-1].state.copy()
                self.env.state_ = x_rnd
                self.env.goal_ = x_goal
                s = get_normal_state(self.env)
                s = np.array([s])
                tmp_samples = self.sampler(s, np.array([Z]))[0].detach().cpu().numpy()
                state_next = recover_state(tmp_samples[0])
                if self.env.valid_state_check(state_next):
                    rnd_node = Node(state_next)
                    good_sample = True

            parent_node = x_rnd_node
            new_node_list = self.steer(parent_node, rnd_node, 2*self.t_propagate, self.t_propagate)
            # parent_node = x_rnd_node
            # rnd_node.parent = parent_node
            # rnd_node.path = [rnd_node.state]
            # new_node_list = [rnd_node]
            # self.node_list += new_node_list

            if len(new_node_list)>0:
                # if self.Eu_metric(new_node_list[-1], self.goal)<5 and not self.env.reach(new_node_list[-1].state, self.goal.state):
                #     new_node_list += self.steer(new_node_list[-1], self.goal, 2*self.t_propagate, self.t_propagate)
                if self.env.reach(new_node_list[-1].state, self.goal.state):
                    near_node = new_node_list[-1]
                    reach_exactly = True
                    if best_node == None or near_node.cost<best_node.cost:
                        best_node = near_node
                    if self.verbose:
                        print('Time: {}'.format(time.time()-tic))
                        print('Path length: {}'.format(best_node.cost))
                    break
        if best_node == None:
            best_node, dis = self.nearest(self.node_list, self.goal, self.se2_metric)
        path = self.generate_final_course(best_node)
        toc = time.time()
        self.planning_time = toc-tic
        self.path = path
        self.reach_exactly = reach_exactly
        return path



class MPNetNoPolicy(MPNet):
    def __init__(self, DeepSampler, n_pcs, **kwargs):
        """
        Args:
            DeepSampler: the trained sampler
            n_pcs: the number of point clouds
            n_limit (int): the number iterations using the deeply informed sampler.
        """
        super().__init__(**kwargs)
        self.DeepSampler = DeepSampler
        self.n_pcs = n_pcs

    def planning(self):
        """
        rrt path planning
        """
        if self.start is None or self.goal is None:
            raise ValueError('start or goal is not set')
        tic = time.time()
        self.node_list = [self.start]
        path = None
        reach_exactly = False
        best_node = None
        
        # normalization factor
        mean_Z = (self.env.workspace_bounds[:,0]+self.env.workspace_bounds[:,1])/2
        scale_Z = (self.env.workspace_bounds[:,1]-self.env.workspace_bounds[:,0])/2
        mean_s = (self.env.state_bounds[:,0]+self.env.state_bounds[:,1])/2
        scale_s = (self.env.state_bounds[:,1]-self.env.state_bounds[:,0])/2

        Z = sample_pcs(self.env, self.n_pcs)
        

        x_rnd = self.start.state.copy()
        x_goal = self.goal.state.copy()

        for i in range(self.max_iter):
            if self.verbose:
                print('Iteration {}: {} nodes'.format(i, len(self.node_list)))

            good_sample = False
            while not good_sample:
                x_rnd_node = self.node_list[-1]
                x_rnd = self.node_list[-1].state.copy()
                if self.env.reach(x_rnd, x_goal):
                    x_rnd_node = self.start
                    x_rnd = self.start.state.copy()
                n_sample = 20
                Z_ = np.zeros((n_sample,np.prod(Z.shape)))+((Z-mean_Z)/scale_Z).reshape((1,-1))
                s = np.block([(x_rnd-mean_s)/scale_s, (x_goal-mean_s)/scale_s])
                s_ = np.zeros((n_sample,len(s)))+s

                tmp_samples = self.DeepSampler(Z_, s_).detach().cpu().numpy()
                tmp_samples = (tmp_samples*scale_s)+mean_s
                
                valid_samples = []
                samples_obs = []
                for k, v in enumerate(tmp_samples):
                    if self.env.valid_state_check(v):
                        valid_samples.append(v)
                        self.env.state_ = v
                        self.env.goal_ = x_goal
                        samples_obs.append(self.env._obs())
                if len(valid_samples) > 0:
                    valid_samples = np.array(valid_samples)
                    samples_obs = np.array(samples_obs)

                    ttr = self.ttr_estimator(samples_obs).detach().cpu().numpy()
                    index = np.argmax(ttr)
                    
                    rnd_node = Node(valid_samples[np.argmin(ttr)])
                    good_sample = True

            
            parent_node = x_rnd_node
            rnd_node.parent = parent_node
            rnd_node.path = [rnd_node.state]
            new_node_list = [rnd_node]
            self.node_list += new_node_list

            if len(new_node_list)>0:
                if self.Eu_metric(new_node_list[-1], self.goal)<5 and not self.env.reach(new_node_list[-1].state, self.goal.state):
                    new_node_list += self.steer(new_node_list[-1], self.goal)
                if self.env.reach(new_node_list[-1].state, self.goal.state):
                    near_node = new_node_list[-1]
                    reach_exactly = True
                    if best_node == None or near_node.cost<best_node.cost:
                        best_node = near_node
                    if self.verbose:
                        print('Time: {}'.format(time.time()-tic))
                        print('Path length: {}'.format(best_node.cost))
                    if not self.optimize:
                        break
        if best_node == None:
            best_node, dis = self.nearest(self.node_list, self.goal, self.se2_metric)
        path = self.generate_final_course(best_node)
        toc = time.time()
        self.planning_time = toc-tic
        self.path = path
        self.reach_exactly = reach_exactly
        return path

class MPNetNaive(MPNet):
    def __init__(self, DeepSampler, n_pcs, **kwargs):
        """
        Args:
            DeepSampler: the trained sampler
            n_pcs: the number of point clouds
            n_limit (int): the number iterations using the deeply informed sampler.
        """
        super().__init__(**kwargs)
        self.DeepSampler = DeepSampler
        self.n_pcs = n_pcs

    def planning(self):
        """
        rrt path planning
        """
        if self.start is None or self.goal is None:
            raise ValueError('start or goal is not set')
        tic = time.time()
        self.node_list = [self.start]
        path = None
        reach_exactly = False
        best_node = None
        
        # normalization factor
        mean_Z = (self.env.workspace_bounds[:,0]+self.env.workspace_bounds[:,1])/2
        scale_Z = (self.env.workspace_bounds[:,1]-self.env.workspace_bounds[:,0])/2
        mean_s = (self.env.state_bounds[:,0]+self.env.state_bounds[:,1])/2
        scale_s = (self.env.state_bounds[:,1]-self.env.state_bounds[:,0])/2

        Z = sample_pcs(self.env, self.n_pcs)
        

        x_rnd = self.start.state.copy()
        x_goal = self.goal.state.copy()

        for i in range(self.max_iter):
            if self.verbose:
                print('Iteration {}: {} nodes'.format(i, len(self.node_list)))

            good_sample = False
            while not good_sample:
                x_rnd_node = self.node_list[-1]
                x_rnd = self.node_list[-1].state.copy()
                if self.env.reach(x_rnd, x_goal):
                    x_rnd_node = self.start
                    x_rnd = self.start.state.copy()
                n_sample = 10
                Z_ = np.zeros((n_sample,np.prod(Z.shape)))+((Z-mean_Z)/scale_Z).reshape((1,-1))
                s = np.block([(x_rnd-mean_s)/scale_s, (x_goal-mean_s)/scale_s])
                s_ = np.zeros((n_sample,len(s)))+s

                tmp_samples = self.DeepSampler(Z_, s_).detach().cpu().numpy()
                tmp_samples = (tmp_samples*scale_s)+mean_s
                
                valid_samples = []
                samples_obs = []
                for k, v in enumerate(tmp_samples):
                    if self.env.valid_state_check(v):
                        valid_samples.append(v)
                        self.env.state_ = v
                        self.env.goal_ = x_goal
                        samples_obs.append(self.env._obs())
                if len(valid_samples) > 0:
                    valid_samples = np.array(valid_samples)
                    samples_obs = np.array(samples_obs)

                    ttr = self.ttr_estimator(samples_obs).detach().cpu().numpy()
                    index = np.argmax(ttr)
                    
                    rnd_node = Node(valid_samples[np.argmin(ttr)])
                    good_sample = True

            
            parent_node = x_rnd_node
            new_node_list = self.steer(parent_node, rnd_node)

            if len(new_node_list)>0:
                if self.Eu_metric(new_node_list[-1], self.goal)<5 and not self.env.reach(new_node_list[-1].state, self.goal.state):
                    new_node_list += self.steer(new_node_list[-1], self.goal)
                if self.env.reach(new_node_list[-1].state, self.goal.state):
                    near_node = new_node_list[-1]
                    reach_exactly = True
                    if best_node == None or near_node.cost<best_node.cost:
                        best_node = near_node
                    if self.verbose:
                        print('Time: {}'.format(time.time()-tic))
                        print('Path length: {}'.format(best_node.cost))
                    if not self.optimize:
                        break
        if best_node == None:
            best_node, dis = self.nearest(self.node_list, self.goal, self.se2_metric)
        path = self.generate_final_course(best_node)
        toc = time.time()
        self.planning_time = toc-tic
        self.path = path
        self.reach_exactly = reach_exactly
        return path

class MPTreeGreedy(MPNet):
    def __init__(self, DeepSampler, n_pcs, **kwargs):
        """
        Args:
            DeepSampler: the trained sampler
            n_pcs: the number of point clouds
            n_limit (int): the number iterations using the deeply informed sampler.
        """
        super().__init__(**kwargs)
        self.DeepSampler = DeepSampler
        self.n_pcs = n_pcs

    def planning(self):
        """
        rrt path planning
        """
        if self.start is None or self.goal is None:
            raise ValueError('start or goal is not set')
        tic = time.time()
        self.node_list = [self.start]
        path = None
        reach_exactly = False
        best_node = None
        
        # normalization factor
        mean_Z = (self.env.workspace_bounds[:,0]+self.env.workspace_bounds[:,1])/2
        scale_Z = (self.env.workspace_bounds[:,1]-self.env.workspace_bounds[:,0])/2
        mean_s = (self.env.state_bounds[:,0]+self.env.state_bounds[:,1])/2
        scale_s = (self.env.state_bounds[:,1]-self.env.state_bounds[:,0])/2

        Z = sample_pcs(self.env, self.n_pcs)
        

        x_rnd = self.start.state.copy()
        x_goal = self.goal.state.copy()

        for i in range(self.max_iter):
            if self.verbose:
                print('Iteration {}: {} nodes'.format(i, len(self.node_list)))

            good_sample = False
            while not good_sample:
                while True:
                    guide_sample = self.sample(self.goal)
                    if self.env.valid_state_check(guide_sample.state):
                        break
                x_rnd_node, _cost = self.nearest(self.node_list, guide_sample, self.ttr_metric)
                x_rnd = x_rnd_node.state
                n_sample = 10
                Z_ = np.zeros((n_sample,np.prod(Z.shape)))+((Z-mean_Z)/scale_Z).reshape((1,-1))
                s = np.block([(x_rnd-mean_s)/scale_s, (x_goal-mean_s)/scale_s])
                s_ = np.zeros((n_sample,len(s)))+s

                tmp_samples = self.DeepSampler(Z_, s_).detach().cpu().numpy()
                tmp_samples = (tmp_samples*scale_s)+mean_s
                
                valid_samples = []
                samples_obs = []
                for k, v in enumerate(tmp_samples):
                    if self.env.valid_state_check(v):
                        valid_samples.append(v)
                        self.env.state_ = v
                        self.env.goal_ = x_goal
                        samples_obs.append(self.env._obs())
                if len(valid_samples) > 0:
                    valid_samples = np.array(valid_samples)
                    samples_obs = np.array(samples_obs)

                    ttr = self.ttr_estimator(samples_obs).detach().cpu().numpy()
                    index = np.argmax(ttr)
                    
                    rnd_node = Node(valid_samples[np.argmin(ttr)])
                    good_sample = True

            
            parent_node = x_rnd_node
            new_node_list = self.steer(parent_node, rnd_node)

            if len(new_node_list)>0:
                if self.Eu_metric(new_node_list[-1], self.goal)<5 and not self.env.reach(new_node_list[-1].state, self.goal.state):
                    new_node_list += self.steer(new_node_list[-1], self.goal)
                if self.env.reach(new_node_list[-1].state, self.goal.state):
                    near_node = new_node_list[-1]
                    reach_exactly = True
                    if best_node == None or near_node.cost<best_node.cost:
                        best_node = near_node
                    if self.verbose:
                        print('Time: {}'.format(time.time()-tic))
                        print('Path length: {}'.format(best_node.cost))
                    if not self.optimize:
                        break
        if best_node == None:
            best_node, dis = self.nearest(self.node_list, self.goal, self.se2_metric)
        path = self.generate_final_course(best_node)
        toc = time.time()
        self.planning_time = toc-tic
        self.path = path
        self.reach_exactly = reach_exactly
        return path

# class MPNet(RRTStar):
#     def __init__(self, DeepSampler, DeepSamplerBackward, n_pcs, **kwargs):
#         """
#         Args:
#             DeepSampler: the trained sampler
#             n_pcs: the number of point clouds
#             n_limit (int): the number iterations using the deeply informed sampler.
#         """
#         super().__init__(**kwargs)
#         self.DeepSampler = DeepSampler
#         self.DeepSamplerBackward = DeepSamplerBackward
#         self.n_pcs = n_pcs

#     def planning(self):
#         """
#         rrt path planning
#         """
#         if self.start is None or self.goal is None:
#             raise ValueError('start or goal is not set')
#         tic = time.time()
#         self.node_list = [self.start]
#         path = None
#         reach_exactly = False
#         best_node = None

#         Z = sample_pcs(self.env, self.n_pcs)

#         success = False

#         # self.bootstrap(20)

#         # if self.reach_exactly:
#         #     return path
#         # path = self.node_list
#         # path = self.node_list[:len(self.node_list)-1] + self.neural_planner(path[-1], self.goal, Z, 100)
#         path = self.neural_planner(self.start, self.goal, Z, 100)
#         depth = 0
#         while not success and depth < 20:
#             path = self.replanning(path, Z)
#             depth += 1
#             if path[-1].parent is not None:
#                 success = True
#             else:
#                 success = False

#         self.node_list = path
#         self.path = path
#         self.reach_exactly = success
#         if not success:
#             for k, v in enumerate(path[1:]):
#                 if v.parent is None:
#                     break
#             self.node_list = self.node_list[:k]
#             self.max_iter = 100
#             self.rrt_planning()
#         toc = time.time()

#         self.planning_time = toc-tic

#         return path

#     def rrt_planning(self):
#         best_node = None
#         reach_exactly = False
#         for i in range(self.max_iter):
#             if self.verbose:
#                 print('Iteration {}: {} nodes'.format(i, len(self.node_list)))
#             good_sample = False
#             while not good_sample:
#                 rnd_node = self.sample(self.goal)
#                 if not self.env.valid_state_check(rnd_node.state): 
#                     continue
#                 nearest_node, cost = self.nearest(self.node_list, rnd_node, self.Eu_metric)
#                 if cost < self.epsilon: 
#                     good_sample = True

#             parent_node = self.choose_parent(rnd_node)
#             new_node_list = self.steer(parent_node, rnd_node)

#             if len(new_node_list)>0 and self.env.reach(new_node_list[-1].state, self.goal.state):
#                 near_node = new_node_list[-1]
#                 reach_exactly = True
#                 if best_node == None or near_node.cost<best_node.cost:
#                     best_node = near_node
#                 if self.verbose:
#                     print('Path length: {}'.format(best_node.cost))
#                 if not self.optimize:
#                     break
#         if best_node == None:
#             best_node, dis = self.nearest(self.node_list, self.goal, self.se2_metric)
#         path = self.generate_final_course(best_node)
#         toc = time.time()
#         self.path = path
#         self.reach_exactly = reach_exactly
#         return path

#     def bootstrap(self, max_iter = 20):
#         """
#         rrt path planning
#         """
#         if self.start is None or self.goal is None:
#             raise ValueError('start or goal is not set')
#         tic = time.time()
#         self.node_list = [self.start]
#         path = None
#         reach_exactly = False
#         best_node = None

#         # normalization factor
#         mean_Z = (self.env.workspace_bounds[:,0]+self.env.workspace_bounds[:,1])/2
#         scale_Z = (self.env.workspace_bounds[:,1]-self.env.workspace_bounds[:,0])/2
#         mean_s = (self.env.state_bounds[:,0]+self.env.state_bounds[:,1])/2
#         scale_s = (self.env.state_bounds[:,1]-self.env.state_bounds[:,0])/2

#         Z = sample_pcs(self.env, self.n_pcs)


#         x_rnd = self.start.state.copy()
#         x_goal = self.goal.state.copy()

#         for i in range(max_iter):
#             if self.verbose:
#                 print('Iteration {}: {} nodes'.format(i, len(self.node_list)))

#             good_sample = False
#             while not good_sample:
#                 x_rnd_node = self.node_list[-1]
#                 x_rnd = self.node_list[-1].state.copy()
#                 if self.env.reach(x_rnd, x_goal):
#                     x_rnd_node = self.start
#                     x_rnd = self.start.state.copy()
#                 n_sample = 10
#                 Z_ = np.zeros((n_sample,np.prod(Z.shape)))+((Z-mean_Z)/scale_Z).reshape((1,-1))
#                 s = np.block([(x_rnd-mean_s)/scale_s, (x_goal-mean_s)/scale_s])
#                 s_ = np.zeros((n_sample,len(s)))+s

#                 tmp_samples = self.DeepSampler(Z_, s_).detach().cpu().numpy()
#                 tmp_samples = (tmp_samples*scale_s)+mean_s

#                 valid_samples = []
#                 samples_obs = []
#                 for k, v in enumerate(tmp_samples):
#                     if self.env.valid_state_check(v):
#                         valid_samples.append(v)
#                         self.env.state_ = v
#                         self.env.goal_ = x_goal
#                         samples_obs.append(self.env._obs())
#                 if len(valid_samples) > 0:
#                     valid_samples = np.array(valid_samples)
#                     samples_obs = np.array(samples_obs)

#                     ttr = self.ttr_estimator(samples_obs).detach().cpu().numpy()
#                     index = np.argmax(ttr)

#                     rnd_node = Node(valid_samples[np.argmin(ttr)])
#                     good_sample = True


#             parent_node = x_rnd_node
#             new_node_list = self.steer(parent_node, rnd_node)

#             if len(new_node_list)>0:
#                 if self.Eu_metric(new_node_list[-1], self.goal)<5 and not self.env.reach(new_node_list[-1].state, self.goal.state):
#                     new_node_list += self.steer(new_node_list[-1], self.goal)
#                 if self.env.reach(new_node_list[-1].state, self.goal.state):
#                     near_node = new_node_list[-1]
#                     reach_exactly = True
#                     if best_node == None or near_node.cost<best_node.cost:
#                         best_node = near_node
#                     if self.verbose:
#                         print('Time: {}'.format(time.time()-tic))
#                         print('Path length: {}'.format(best_node.cost))
#                     if not self.optimize:
#                         break
#         if best_node == None:
#             best_node, dis = self.nearest(self.node_list, self.goal, self.se2_metric)
#         path = self.generate_final_course(best_node)
#         toc = time.time()
#         self.planning_time = toc-tic
#         self.path = path
#         self.reach_exactly = reach_exactly
#         return path

#     def steerto(self, from_node, to_node, duration=5.0):
#         """Steer the start node to goal node
#         Args:
#             start: Node
#             goal: Node
#             duration: float
#         Returns:
#             The node extended in the maximum distance.
#         """
#         parent_node = from_node
#         new_node = Node(from_node.state)
#         new_node.parent = parent_node
#         new_node.path.append(new_node.state)
#         state = new_node.state
#         goal = to_node.state.copy()

#         env = self.env
#         env.state_ = state
#         env.goal_ = goal

#         dt = env.step_dt
#         n_max_extend = round(duration/dt)
#         n_extend = 0
#         while not (n_extend>n_max_extend or env.reach(new_node.state, goal) or not env.valid_state_check(new_node.state)):
#             obs = env._obs()
#             action, _ = self.policy.actor(obs.reshape(1,-1), eps=0.01)
#             action = action.detach().cpu().numpy()[0]
#             obs, rewards, done, info = env.step(action)

#             new_node.state = env.state_
#             new_node.path.append(new_node.state)
#             n_extend += 1    
#         return new_node

#     def neural_planner(self, start, goal, Z, max_step):
#         """Plan a path connecting the start and goal, but do not check its viability
#         Args:
#             start: node
#             goal: node
#             Z: pcs
#             max_step: int

#         Returns:
#             path: a list of nodes
#         """
#         tau_a = [start]
#         tau_b = [goal]
#         reached = False
#         step = 0
#         direction = 1
#         while not reached and step < max_step:
#             if direction == 1:
#                 x_new = self.deep_sampling(self.DeepSampler, tau_a[-1], tau_b[-1], Z)
#                 tau_a += [x_new]
#             #     direction = 0
#             # else:
#             #     x_new = self.deep_sampling(self.DeepSamplerBackward, tau_b[-1], tau_a[-1], Z, n_batch=1)
#             #     tau_b += [x_new]
#             #     direction = 1
#             step += 1
#             if self.Eu_metric(tau_a[-1], tau_b[-1])<5:
#                 node = self.steerto(tau_a[-1], tau_b[-1])
#                 if self.env.valid_state_check(node.state) and self.env.reach(node.state, tau_b[-1].state):
#                     reached = True
#         tau_b.reverse()
#         return tau_a + tau_b

#     def replanning(self, path, Z):
#         """
#         Args:
#             path: a list of nodes
#             Z: point clouds
#         Returns:
#             new_path: a list of nodes
#         """
#         new_path = [path[0]]
#         i = 1
#         while i < len(path):
#             start = new_path[-1]
#             goal = path[i]
#             if goal.parent == start:
#                 new_path.append(goal)
#                 i += 1
#             else:
#                 near_nodes = self.near(path[i:], start, self.Eu_metric, 10.0)
#                 suc = False
#                 if near_nodes is not None: # near nodes exist
#                     # sort the near nodes with their Q values
#                     samples_obs = []
#                     for k, v in enumerate(near_nodes):
#                         self.env.state_ = v.state
#                         self.env.goal_ = path[-1].state
#                         samples_obs.append(self.env._obs())
#                     samples_obs = np.array(samples_obs)
#                     ttr = self.ttr_estimator(samples_obs).detach().cpu().numpy()
#                     index_array = np.argsort(ttr) # ttr ascend

#                     for ind in index_array:
#                         node = self.steerto(start, near_nodes[ind], 10)   
#                         if self.env.valid_state_check(node.state) and self.env.reach(node.state, near_nodes[ind].state):
#                             new_path.append(node)
#                             i = path.index(near_nodes[ind]) + 1
#                             suc = True
#                             break
#                 else:
#                     node = self.steerto(start, goal, 10)   
#                     if self.env.valid_state_check(node.state) and self.env.reach(node.state, goal.state):
#                         new_path.append(node)
#                         i = i + 1
#                         suc = True

#                 if not suc:
#                     mid_path = self.neural_planner(start, goal, Z, 20)
#                     new_path += mid_path[1:]
#                     if i+1<len(path):
#                         new_path += path[i+1:]
#                     break
#         return new_path

#     def deep_sampling(self, DeepSampler, start, goal, Z, n_batch = 10):
#         """
#         Args: 
#             start: node
#             goal: node
#             Z: pcs
#             n_batch: int

#         Returns:
#             rnd_node: a random sample node
#         """
#         start = start.state
#         goal = goal.state
#         # normalization factor
#         mean_Z = (self.env.workspace_bounds[:,0]+self.env.workspace_bounds[:,1])/2
#         scale_Z = (self.env.workspace_bounds[:,1]-self.env.workspace_bounds[:,0])/2
#         mean_s = (self.env.state_bounds[:,0]+self.env.state_bounds[:,1])/2
#         scale_s = (self.env.state_bounds[:,1]-self.env.state_bounds[:,0])/2
#         while True:
#             Z_ = np.zeros((n_batch,np.prod(Z.shape)))+((Z-mean_Z)/scale_Z).reshape((1,-1))
#             s = np.block([(start-mean_s)/scale_s, (goal-mean_s)/scale_s])
#             s_ = np.zeros((n_batch,len(s)))+s

#             tmp_samples = DeepSampler(Z_, s_).detach().cpu().numpy()
#             tmp_samples = (tmp_samples*scale_s)+mean_s

#             valid_samples = []
#             samples_obs = []
#             for k, v in enumerate(tmp_samples):
#                 if self.env.valid_state_check(v):
#                     valid_samples.append(v)
#                     self.env.state_ = v
#                     self.env.goal_ = goal
#                     samples_obs.append(self.env._obs())
#             if len(valid_samples) > 0:
#                 valid_samples = np.array(valid_samples)
#                 samples_obs = np.array(samples_obs)

#                 ttr = self.ttr_estimator(samples_obs).detach().cpu().numpy()

#                 rnd_node = Node(valid_samples[np.argmin(ttr)])
#                 return rnd_node

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_num', type=int, default=1000)
    parser.add_argument('--logdir', type = str, default='log')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--step_size', type=float, default=1.0)
    return parser.parse_known_args()[0]


def test(args = get_args()):
    env = CarEnv()

    # sampler
    sampler = SamplerImage(env.observation_space['dynamics'].shape, 3, [-1, 1], env.observation_space['local_map'].shape, 64, device=args.device).to(args.device)

    sampler.load_state_dict(torch.load(os.path.join(
                args.logdir+'/sst_sampler', 'policy_200.pth'), map_location=args.device))

    # steer
    policy = Actor(None, env.observation_space['dynamics'].shape, env.action_space.shape, [-1, 1], args.device).to(args.device)
    policy.load_state_dict(torch.load(os.path.join(
                args.logdir+'/sst', 'policy_500.pth'), map_location=args.device))
        
    planner = MPNet(env, policy, sampler, max_iter=300, verbose=False)

    save_dir = args.logdir + '/dubin_path/env1'
    suc = []
    planning_time = []
    for i in range(100):
        data = pickle.load(open(f'{save_dir}/path{i}.pkl', 'rb'))
        _states = data['states']
        _controls = data['controls']
        _local_maps = data['local_maps']

        start = _states[0]
        goal = _states[-1]
        planner.set_start_and_goal(start, goal)
        planner.planning()
        suc.append(planner.reach_exactly)
        planning_time.append(planner.planning_time)
        save_data = {
            'node_list': planner.node_list,
            'path': planner.path,
            'start': start,
            'goal': goal,
            'sst_path': _states
        }
        pickle.dump(save_data, open(f'{args.logdir}/mpnet_results/path{i}.pkl', 'wb'))
    pickle.dump(np.array(suc), open(f'{args.logdir}/mpnet_results/suc.pkl', 'wb'))
    pickle.dump(np.array(planning_time), open(f'{args.logdir}/mpnet_results/planning_time.pkl', 'wb'))

def test_pcs(args = get_args()):
    env = CarEnv()

    pcs_num = 4000
    # sampler
    sampler = SamplerPcs(env.observation_space['dynamics'].shape, 3, [-1,1], pcs_num*2, 64, device=args.device).to(args.device)

    sampler.load_state_dict(torch.load(os.path.join(
                args.logdir+'/sst_sampler', 'pcs_policy_200.pth'), map_location=args.device))

    # steer
    policy = Actor(None, env.observation_space['dynamics'].shape, env.action_space.shape, [-1, 1], args.device).to(args.device)
    policy.load_state_dict(torch.load(os.path.join(
                args.logdir+'/sst', 'policy_500.pth'), map_location=args.device))
        
    planner = MPNetPcs(env, policy, sampler, max_iter=300, verbose=False)

    save_dir = args.logdir + '/dubin_path/env1'
    suc = []
    planning_time = []
    for i in range(100):
        data = pickle.load(open(f'{save_dir}/path{i}.pkl', 'rb'))
        _states = data['states']
        _controls = data['controls']
        _local_maps = data['local_maps']

        start = _states[0]
        goal = _states[-1]
        planner.set_start_and_goal(start, goal)
        planner.planning()
        suc.append(planner.reach_exactly)
        planning_time.append(planner.planning_time)
        save_data = {
            'node_list': planner.node_list,
            'path': planner.path,
            'start': start,
            'goal': goal,
            'sst_path': _states
        }
        pickle.dump(save_data, open(f'{args.logdir}/mpnet_results/path{i}.pkl', 'wb'))
    pickle.dump(np.array(suc), open(f'{args.logdir}/mpnet_results/suc.pkl', 'wb'))
    pickle.dump(np.array(planning_time), open(f'{args.logdir}/mpnet_results/planning_time.pkl', 'wb'))


if __name__ == "__main__":
    test()
    # test_pcs()
