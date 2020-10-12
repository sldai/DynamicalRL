import random
import numpy as np
import time
from rl_planner.planner.base_rrt import BaseRRT, Node


class RRT(BaseRRT):
    """
    RRT using time to reach as the cost. It always tries to extend the node with the lowest heuristic to the sample.
    """
    def __init__(self, env, policy, ttr_estimator, goal_sample_rate=5, max_iter=500, **kwargs):
        """
        """
        super().__init__(env=env, **kwargs)

        self.ttr_estimator = ttr_estimator
        self.policy = policy

        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.epsilon = 10.0 # near radius


    def planning(self, verbose=True):
        """Path planning,
        """
        if self.start is None or self.goal is None:
            raise ValueError('start or goal is not set')
        tic = time.time()
        self.node_list = [self.start]
        path = None
        reach_exactly = False
        for i in range(self.max_iter):
            if verbose:
                print('Iteration {}: {} nodes'.format(i, len(self.node_list)))
            good_sample = False
            while not good_sample:
                rnd_node = self.sample(self.goal)
                if not self.env.valid_state_check(rnd_node.state): 
                    continue
                nearest_node, cost = self.nearest(self.node_list, rnd_node, self.Eu_metric)
                if cost < self.epsilon: 
                    good_sample = True
            
            parent_node = self.choose_parent(rnd_node)
            new_node_list = self.steer(parent_node, rnd_node)

            if len(new_node_list)>0:
                if self.nearest(new_node_list, self.goal, self.se2_metric)[1] <= 1.5: # reach goal
                    reach_exactly = True
                    break
                nearest_node, cost = self.nearest(self.node_list, self.goal, self.Eu_metric)
                if cost < 5.0: 
                    parent_node = self.choose_parent(self.goal)
                    new_node_list = self.steer(parent_node, self.goal)
                    if len(new_node_list)>0 and self.nearest(new_node_list, self.goal, self.se2_metric)[1] <= 1.5:
                        reach_exactly = True
                        break
        path = self.generate_final_course(self.node_list[-1])
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

    def ttr_metric(self, src_node, dst_node):
        self.env.state_ = src_node.state
        self.env.goal_ = dst_node.state
        obs = self.env._obs().reshape(1,-1)
        ttr = self.ttr_estimator(obs).item()
        return ttr

    def choose_parent(self, rnd_node):
        """The chosen node has the lowest heuristic.
        """
        near_nodes = self.near(self.node_list, rnd_node, self.Eu_metric, self.epsilon)
        parent_node = None
        if near_nodes == None:
            parent_node, min_c_cost = self.nearest(self.node_list, rnd_node, self.Eu_metric)
        else:
            parent_node, min_h_cost = self.nearest(near_nodes, rnd_node, self.ttr_metric)
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
            obs = env._obs()
            action, _ = self.policy.actor(obs.reshape(1,-1), eps=0.01)
            action = action.detach().cpu().numpy()[0]
            obs, rewards, done, info = env.step(action)

            new_node.state = env.state_
            new_node.path.append(new_node.state)
            n_extend += 1
            if not info['collision'] and (n_extend % n_tree == 0 or info['goal']):
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
                node.cost = parent_node.cost+0.2*(len(node.path)-1) 
                self.propagate_cost_to_leaves(node)
  
