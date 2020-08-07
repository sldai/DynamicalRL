import pickle
from matplotlib import pyplot as plt

import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import VectorEnv
from tianshou.policy import DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, Batch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../underactuated_envs/")
from quadcopter import QuadcopterEnv
from net import Actor, Critic



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help='train or test a policy')
    parser.add_argument('--task', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer_size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=2400)
    parser.add_argument('--collect-per-step', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    # parser.add_argument('--layer', type=list, default=[1024, 512, 512, 512])
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args

from rl_planner.env.quadcopter import QuadcopterEnv
def gym_make():
    
    # obs_list_list = pickle.load(open(os.path.dirname(
    #     __file__)+'/../data/obstacles/obs_list_list.pkl', 'rb'))
    # training_env = pickle.load(open(os.path.dirname(__file__)+'/../data/obstacles/test_env2.pkl', 'rb'))
    # env = CartPoleEnv()
    # obs1 = RectObs(rect=np.array([[-6,-4],[-4,-2]],dtype=float))
    # obs2 = RectObs(rect=np.array([[-2,2],[2,4]],dtype=float))
    # obs3 = RectObs(rect=np.array([[4,-4],[6,-2]],dtype=float))
    # env.set_obs([])
    env = QuadcopterEnv()
    return env


def train(args=get_args()):
    torch.set_num_threads(1)  # we just need only one thread for NN
    env = gym_make()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.action_range = [env.action_space.low[0], env.action_space.high[0]]
    args.layer = [1024, 512, 512, 512]

    train_envs = VectorEnv(
        [lambda: gym_make() for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = VectorEnv(
        [lambda: gym_make() for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    actor = Actor(
        args.layer, args.state_shape, args.action_shape,
        args.action_range, args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic = Critic(
        args.layer, args.state_shape, args.action_shape, args.device
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        args.tau, args.gamma, args.exploration_noise,
        args.action_range,
        reward_normalization=args.rew_norm, ignore_done=True)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ddpg')
    writer = SummaryWriter(log_path)

    # if a model exist, continue to train it
    model_path = os.path.join(log_path, 'policy.pth')
    # if os.path.exists(model_path):
    #     policy.load_state_dict(torch.load(model_path))
    def save_fn(policy):
        torch.save(policy.state_dict(), model_path)


    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, save_fn=save_fn, writer=writer)
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        # Let's watch its performance!
        env = gym_make()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


def test(args=get_args()):
    torch.set_num_threads(1)  # we just need only one thread for NN

    env = gym_make()

    model_path = os.path.join(args.logdir, args.task, 'ddpg/policy.pth')
    layer = [1024, 512, 512, 512]
    device = 'cuda'

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    action_range = [env.action_space.low, env.action_space.high]
    actor = Actor(
        layer, state_shape, action_shape,
        action_range, device
    ).to(device)
    critic = Critic(
        layer, state_shape, action_shape, device
    ).to(device)

    actor = actor.to(device)
    actor_optim = torch.optim.Adam(actor.parameters())
    critic = critic.to(device)
    critic_optim = torch.optim.Adam(critic.parameters())
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        action_range=action_range)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    obs = env.reset()
    # env.state[0] = -30.0
    # env.goal[0] = 30.0
    env.render()
    print(env.goal)
    while True:
        action, _ = policy.actor(obs.reshape(1,-1), eps=0.01)
        action = action.detach().cpu().numpy()[0]
        
        obs, reward, done, info = env.step(action)
        # print(env.state)
        # print(reward)
        # print(info)
        env.render()
        if done:
            break


if __name__ == '__main__':
    args = get_args()
    if args.train:
        train(args)
    else:
        test(args)
        plt.show()
