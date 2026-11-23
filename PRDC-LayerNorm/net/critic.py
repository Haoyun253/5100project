import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAffine(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        return x * self.weight + self.bias


def hidden_transform(mode, depth):
    if mode == "ln-no-affine":
        return nn.LayerNorm(256, elementwise_affine=False)
    if mode == "affine-only":
        return FeatureAffine(256)
    layers = {"layernorm": (1, 2), "ln-first": (1,), "ln-second": (2,)}
    return nn.LayerNorm(256) if depth in layers[mode] else nn.Identity()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mode="layernorm"):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.q1_norm1 = hidden_transform(mode, 1)
        self.q1_norm2 = hidden_transform(mode, 2)
        self.q2_norm1 = hidden_transform(mode, 1)
        self.q2_norm2 = hidden_transform(mode, 2)

    def _q1(self, sa):
        q1 = F.relu(self.q1_norm1(self.l1(sa)))
        q1 = F.relu(self.q1_norm2(self.l2(q1)))
        return self.l3(q1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self._q1(sa)
        q2 = F.relu(self.q2_norm1(self.l4(sa)))
        q2 = F.relu(self.q2_norm2(self.l5(q2)))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self._q1(sa)
