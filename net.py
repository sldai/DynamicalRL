import torch
import numpy as np
from torch import nn


class Actor(nn.Module):
    def __init__(self, layer, state_shape, action_shape,
                 action_range, device='cpu'):
        """
        Parameters
        ----------
        list layer: [int, int, ...] 
            network hidden layer
        """

        super().__init__()
        self.device = device
        if layer is None:
            layer = [512, 512, 128]
        self.model = [
            nn.Linear(np.prod(state_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(layer[-1], np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self.low = torch.tensor(action_range[0], device=self.device)
        self.high = torch.tensor(action_range[1], device=self.device)

        self.action_bias = (self.low+self.high)/2
        self.action_scale = (self.high-self.low)/2

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        # logits = torch.tanh(logits)
        # if kwargs.get('eps') is not None:
        #     eps = kwargs['eps']
        #     logits = logits + torch.randn(
        #             size=logits.shape, device=logits.device) * eps
        # # scale the logits to produce actions
        # logits = logits * self.action_scale.view((1,-1)) + self.action_bias.view((1,-1))
        # logits = torch.min(torch.max(logits, self.low.view((1,-1))), self.high.view((1,-1)))
        return logits, None


class ActorProb(nn.Module):
    def __init__(self, layer, state_shape, action_shape,
                 action_range, device='cpu'):
        super().__init__()
        if layer is None:
            layer = [512, 512, 128]
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(layer[-1], np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class Critic(nn.Module):
    def __init__(self, layer, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        if layer is None:
            layer = [512, 512, 128]
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(layer[-1], 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits

class Conv(nn.Module):
    def __init__(self, input_size, output_size, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Conv2d(1,8,kernel_size=(3,3), padding=1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.PReLU(),
            nn.Conv2d(8,16,kernel_size=(3,3), padding=1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.PReLU(),
            nn.Conv2d(16,32,kernel_size=(3,3), padding=1),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(2)
        ]
        self.model = nn.Sequential(*self.model)
        rand_sample = torch.rand((1,)+input_size)
        length = list(self.model(rand_sample).flatten().size())[0]
        print(length)
        self.linear = nn.Linear(length, output_size)

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        logits = self.model(s)
        logits = torch.flatten(logits, 1)
        logits = self.linear(logits)
        return logits

class SamplerPcs(nn.Module):
    def __init__(self, state_shape, action_shape,
                 action_range, map_feature_size, hidden_feature_size, device='cpu'):
        """
        Parameters
        ----------
        list layer: [int, int, ...] 
            network hidden layer
        """
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(map_feature_size, 256), nn.PReLU(),
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, hidden_feature_size))

        self.mlp = nn.Sequential(
            nn.Linear(np.prod(state_shape)+hidden_feature_size, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(),
            nn.Linear(64, np.prod(action_shape)))

        self.low = torch.tensor(action_range[0], device=self.device)
        self.high = torch.tensor(action_range[1], device=self.device)

        self.action_bias = (self.low+self.high)/2
        self.action_scale = (self.high-self.low)/2

    def forward(self, s, pcs, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if not isinstance(pcs, torch.Tensor):
            pcs = torch.tensor(pcs, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        hidden_features = self.encoder(pcs)
        logits = torch.cat((s,hidden_features), 1)
        logits = self.mlp(logits)
        return logits, None

class SamplerImage(nn.Module):
    def __init__(self, state_shape, action_shape,
                 action_range, image_shape, hidden_feature_size, device='cpu'):
        """
        Parameters
        ----------
        list layer: [int, int, ...] 
            network hidden layer
        """
        super().__init__()
        self.device = device
        self.conv = Conv(image_shape, hidden_feature_size)

        self.mlp = nn.Sequential(
            nn.Linear(np.prod(state_shape)+hidden_feature_size, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(),
            nn.Linear(64, np.prod(action_shape)))

        self.low = torch.tensor(action_range[0], device=self.device)
        self.high = torch.tensor(action_range[1], device=self.device)

        self.action_bias = (self.low+self.high)/2
        self.action_scale = (self.high-self.low)/2

    def forward(self, s, image, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        hidden_features = self.conv(image)
        logits = torch.cat((s,hidden_features), 1)
        logits = self.mlp(logits)
        return logits, None

