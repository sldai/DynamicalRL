from rl_planner.planner.rrt import RRT
import time
import numpy as np


class RRTStar(RRT):
    def __init__(self, optimize=True, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.optimize = optimize
        self.verbose = verbose

    def planning(self, ):
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

        for i in range(self.max_iter):
            if self.verbose:
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

            if len(new_node_list)>0 and self.env.reach(new_node_list[-1].state, self.goal.state):
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
        

    def choose_parent(self, rnd_node):
        """
        Choose the node to extend after sampling a random node,
        the chosen node has the lowest cost, i.e. h=g+c 
        """
        near_nodes = self.near(self.node_list, rnd_node, self.Eu_metric, self.epsilon)
        parent_node = None
        if near_nodes == None:
            parent_node, min_c_cost = self.nearest(self.node_list, rnd_node, self.Eu_metric)
        else:
            costs = np.array([node.cost + self.ttr_metric(node, rnd_node) for node in near_nodes])
            min_ind = np.argmin(costs)
            parent_node, min_h_cost = near_nodes[min_ind], costs[min_ind]
        return parent_node
