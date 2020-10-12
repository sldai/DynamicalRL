import torch
import pickle
from net import Actor, Conv, SamplerImage, SamplerPcs
import argparse

import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../DynamicalEnvs/")
from car import CarEnv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='train_pcs')
    parser.add_argument('--traj_num', type=int, default=1000)
    parser.add_argument('--logdir', type = str, default='log')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--step_size', type=float, default=1.0)
    return parser.parse_known_args()[0]

from torch import nn
from torch.utils.tensorboard import SummaryWriter

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

def train_pcs(arg=get_args()):
    env = CarEnv()
    states = []
    acts = []
    total_pcs = []
    pcs_index = []
    pcs_num = 4000

    step_size = int(args.step_size/env.step_dt)
    save_dir = args.logdir + '/dubin_path/env1'
    for i in range(args.traj_num):
        data = pickle.load(open(f'{save_dir}/path{i}.pkl', 'rb'))
        _states = data['states']
        _states[:,0] = _states[:,0]/20
        _states[:,1] = _states[:,1]/20
        _states[:,2] = _states[:,2]/np.pi
        _states[:,3] = _states[:,3]/env.state_bounds[3,1]
        _controls = data['controls']
        _local_maps = data['local_maps']
        tmp_pcs = sample_pcs(env, pcs_num)
        # normalize
        tmp_pcs[:,0]/=20
        tmp_pcs[:,1]/=20
        total_pcs.append(tmp_pcs.reshape((pcs_num*2,)))
        for ind in range(len(_states)-1):
            start_ind = ind
            goal_ind = len(_states)-1
            action_ind = ind+step_size if ind+step_size<len(_states) else len(_states)-1
            state = _states[start_ind]
            goal = _states[goal_ind]
            states.append([state[0]-goal[0], state[1]-goal[1], state[2], state[3], goal[2]])
            next_state = _states[action_ind] 
            acts.append(next_state[:3])
            pcs_index.append(len(total_pcs)-1)

            
    states = np.array(states)
    acts = np.array(acts)
    total_pcs = np.array(total_pcs)
    pcs_index = np.array(pcs_index, dtype=int)
    
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
    total_pcs = torch.tensor(
        total_pcs, device=args.device, dtype=torch.float)
    pcs_index = pcs_index[indices]

    model = SamplerPcs(len(states[0]), len(acts[0]), [-1,1], pcs_num*2, 64, device=args.device).to(args.device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters())

    # Train the Models

    best_epoch = None
    best_loss = None
    args.model_path = os.path.join(args.logdir, 'sst_sampler')
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
            b_states, b_acts, b_total_pcs = states[i:i_], acts[i:i_], total_pcs[pcs_index[i:i_]]

            c_acts, _ = model(b_states, b_total_pcs)
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
                    args.model_path, f'pcs_policy_{epoch}.pth'))
    writer.close()

def train(args=get_args()):
    env = CarEnv()
    states = []
    acts = []
    local_maps = []
    step_size = int(args.step_size/env.step_dt)
    save_dir = args.logdir + '/dubin_path/env1'
    for i in range(args.traj_num):
        data = pickle.load(open(f'{save_dir}/path{i}.pkl', 'rb'))
        _states = data['states']
        _states[:,0] = _states[:,0]/20
        _states[:,1] = _states[:,1]/20
        _states[:,2] = _states[:,2]/np.pi
        _states[:,3] = _states[:,3]/env.state_bounds[3,1]
        _controls = data['controls']
        _local_maps = data['local_maps']
        for ind in range(len(_states)-1):
            start_ind = ind
            goal_ind = len(_states)-1
            action_ind = ind+step_size if ind+step_size<len(_states) else len(_states)-1
            state = _states[start_ind]
            goal = _states[goal_ind]
            states.append([state[0]-goal[0], state[1]-goal[1], state[2], state[3], goal[2]])
            next_state = _states[action_ind] 
            acts.append(next_state[:3])
            local_maps.append(_local_maps[start_ind])
    states = np.array(states)
    acts = np.array(acts)
    local_maps = np.array(local_maps)
    
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
    local_maps = torch.tensor(
        local_maps[indices], device=args.device, dtype=torch.float)
    model = SamplerImage(len(states[0]), len(acts[0]), [-1, 1], env.observation_space['local_map'].shape, 64, device=args.device).to(args.device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters())

    # Train the Models

    best_epoch = None
    best_loss = None
    args.model_path = os.path.join(args.logdir, 'sst_sampler')
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
            b_states, b_acts, b_local_maps = states[i:i_], acts[i:i_], local_maps[i:i_]
            c_acts, _ = model(b_states, b_local_maps)
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
                    args.model_path, f'policy_{epoch}.pth'))
    writer.close()

def test(args=get_args()):
    env = CarEnv()
    env.max_time = 5.0
    # env.set_obs([])
    model = Actor(None, env.observation_space['dynamics'].shape, env.action_space.shape, [-1, 1], args.device).to(args.device)
    args.model_path = os.path.join(args.logdir, 'sst')
    model.load_state_dict(torch.load(os.path.join(
                    args.model_path, 'policy.pth'), map_location=args.device))
    for i in range(10):
        env.reset(low=2,high=4)
        # env.state[:2] -= env.goal[:2]
        # env.goal[:2] -= env.goal[:2]
        obs = env._obs()
        env.render()
        done = False
        while not done:
            normed_obs = obs['dynamics'].reshape((1,-1))
            obs = np.array([env.state[0]-env.goal[0], env.state[1]-env.goal[1], env.state[2], env.state[3], env.goal[2]])
            # /np.array([20,20,np.pi,1,np.pi])

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
    elif args.case == 'train_pcs':
        train_pcs()
