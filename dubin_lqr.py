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
from dubin import DubinEnv

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

from torch import nn
from torch.utils.tensorboard import SummaryWriter

def train(args=get_args()):
    trajs = pickle.load(open('dubin_training_data.pkl', 'rb'))
    states, acts = trajs['obs'], trajs['action']
    
    env = DubinEnv()
    # states[:,0] = states[:,0]/20
    # states[:,1] = states[:,1]/20
    # states[:,2] = states[:,2]/np.pi
    # # states[:,3] = states[:,3]/env.state_bounds[3,1]
    # states[:,4] = states[:,4]/np.pi
    acts = np.clip(env.normalize_u(acts), -1, 1)
    print(np.max(abs(states[:,0])), np.max(abs(states[:,1])), np.max(abs(states[:,2])), np.max(abs(acts[:,0])), np.max(abs(acts[:,1])))
    print(states[-1], acts[-1])
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

    model = Actor(None, env.observation_space['dynamics'].shape, env.action_space.shape, [-1, 1], args.device).to(args.device)

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
    env = DubinEnv()
    env.set_obs([])
    model = Actor(None, env.observation_space['dynamics'].shape, env.action_space.shape, [-1, 1], args.device).to(args.device)
    args.model_path = os.path.join(args.logdir, 'lqr')
    model.load_state_dict(torch.load(os.path.join(
                    args.model_path, 'policy.pth'), map_location=args.device))
    for i in range(10):
        env.reset()
        env.state[:2] -= env.goal[:2]
        env.goal[:2] -= env.goal[:2]
        obs = env._obs()
        env.render()
        done = False
        while not done:
            normed_obs = obs['dynamics'].reshape((1,-1))
            # /np.array([20,20,np.pi,1,np.pi])

            act = model(normed_obs)[0].detach().cpu().numpy()[0]
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