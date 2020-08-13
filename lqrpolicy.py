import torch
import pickle
from net import Actor
from scipy.linalg import solve_continuous_are, solve_discrete_are
import argparse

import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../DynamicalEnvs/")
from quadcopter import QuadcopterEnv, mass, g, Ix, Iy, Iz


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='collect')
    parser.add_argument('--traj_num', type=int, default=1000)
    parser.add_argument('--logdir', type = str, default='log')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=500)
    return parser.parse_known_args()[0]


def LQR_discrete_gain(dt):
    A = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    A = np.eye(12) + A*dt

    B = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1/mass, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1/Ix, 0, 0],
        [0, 0, 1/Iy, 0],
        [0, 0, 0, 1/Iz],
    ])

    B = B*dt

    # Q = np.eye(12)*1.0
    Q = np.diag([1, 1.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    R = np.eye(4)*0.001

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


def LQR_control(env):
    """Collect a trajectory using LQR.
    Args:
        env: the gym environment

    Returns:
        states: states during the trajectory
        acts: actions during the trajectory
    """
    K = LQR_discrete_gain(env.dt)
    x_e, u_e = env.goal_, np.array([mass*g, 0, 0, 0])  # equalibrium

    states = []
    acts = []

    done = False
    while not done:
        obs = env._obs()
        states.append(obs)
        u = u_e - K @ (env.state_-x_e)
        act = env.normalize_u(u)
        acts.append(act)
        obs, reward, done, info = env.step(act)   
        
    return states, acts


def collect_data(args=get_args()):
    env = QuadcopterEnv()

    states, acts = [], []
    for i in range(1, 1+args.traj_num):
        env.reset()
        _states, _acts = LQR_control(env)
        states += _states
        acts += _acts
        if i % 100 == 0:
            print(f'Generate {i} trajectories')

    state_act_pair = {'states': np.array(states), 'acts': np.array(acts)}
    pickle.dump(state_act_pair, open('lqr_quadcopter', 'wb'))

from torch import nn
from torch.utils.tensorboard import SummaryWriter

def train(args=get_args()):
    trajs = pickle.load(open('lqr_quadcopter', 'rb'))
    states, acts = trajs['states'], trajs['acts']
    env = QuadcopterEnv()
    assert len(states) == len(acts), 'X, y sizes mismatch'
    print(len(states))
    # shuffle
    indices = np.arange(len(states), dtype=int)
    np.random.shuffle(indices)
    # convert training data into tensors
    states = torch.tensor(
        states[indices], device=args.device, dtype=torch.float)
    acts = torch.tensor(
        acts[indices], device=args.device, dtype=torch.float)

    model = Actor(None, env.observation_space.shape, env.action_space.shape, [-1, 1], args.device).to(args.device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters())

    # Train the Models

    best_epoch = None
    best_loss = None
    args.model_path = os.path.join(args.logdir, 'lqr')
    writer = SummaryWriter(log_dir=args.model_path)

    for epoch in range(1, args.epoch+1):
        loss_train = 0.0
        for i in range(0, len(states), args.batch_size):
            # Forward, Backward and Optimize

            # we need zero_grad since pytorch accumulates gradient
            # this is useful in weight sharing like CNN
            model.zero_grad()

            i_ = i+args.batch_size if i + \
                args.batch_size <= len(states) else len(states)
            b_states, b_acts = states[i:i_], acts[i:i_]
            c_acts = model(b_states)[0]
            loss_ = loss(c_acts, b_acts)

            loss_.backward()
            optimizer.step()

            loss_train = loss_train+loss_
        # loss_train = loss(model(total_pcs[maps],states), nexts)
        writer.add_scalar('Loss/train', loss_train /
                          (len(states)/args.batch_size), epoch)

        # Save the models
        if epoch != 0 and epoch % 50 == 0:
            print("loss: {} in #{}".format(loss_train, epoch))
            if best_epoch == None or loss_train < best_loss:
                best_loss, best_epoch = loss_train, epoch
            print('best_loss: {} in #{}'.format(best_loss, best_epoch))
            if best_epoch == epoch:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'policy.pth'))
    writer.close()

def test(args=get_args()):
    env = QuadcopterEnv()
    model = Actor(None, env.observation_space.shape, env.action_space.shape, [-1, 1], args.device).to(args.device)
    args.model_path = os.path.join(args.logdir, 'lqr')
    model.load_state_dict(torch.load(os.path.join(
                    args.model_path, 'policy.pth'), map_location=args.device))
    for i in range(10):
        obs = env.reset()
        env.render()
        done = False
        while not done:
            act = model(obs.reshape((1,-1)))[0].detach().cpu().numpy()[0]
            obs, reward, done, info = env.step(act)
            env.render()



if __name__ == "__main__":
    args = get_args() 
    if args.case == 'collect':
        collect_data()
    elif args.case == 'train':
        train()
    elif args.case == 'test':
        test()
