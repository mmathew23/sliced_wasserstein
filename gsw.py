import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def normalize_weights(weights):
    return weights / torch.norm(weights, dim=1, keepdim=True)


class Projector(nn.Module):
    """
        Base Class for linear and non-linear projections
        freeze and unfreeze
        reset params
    """
    def __init__(self):
        super(Projector, self).__init__()
        self.training = False

    def set_training(self, status):
        self.training = status
        for param in self.parameters():
            param.requires_grad = self.training

    def reset(self, force=False):
        return NotImplemented


class LinearProjector(Projector):
    def __init__(self, input_features, final_dim):
        super(LinearProjector, self).__init__()
        self.projection = nn.Parameter(torch.randn(final_dim, input_features))
        self.reset()
        self.set_training(False)

    def reset(self, force=False):
        if not self.training or force:
            self.projection.data = torch.randn_like(self.projection)

    def forward(self, x):
        return F.linear(x, normalize_weights(self.projection), None)


class NNProjector(Projector):
    def __init__(
        self, num_layer, input_features, hidden_dim, final_dim,
    ):
        super(NNProjector, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_features, hidden_dim, bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        for i in range(1, num_layer):
            if i < num_layer-1:
                self.projection.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=False)
                )
                self.projection.append(nn.LeakyReLU(negative_slope=0.2))
            else:
                self.final_projection = nn.Parameter(
                    torch.randn(final_dim, hidden_dim)
                )
        self.reset()
        self.set_training(False)

    def reset(self, force=False):
        if not self.training or force:
            for module in self.projection.children():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(
                        module.weight, gain=nn.init.calculate_gain('leaky_relu')
                    )
            self.final_projection.data = torch.randn_like(
                self.final_projection
            )

    def forward(self, x):
        x = self.projection(x)
        return F.linear(x, normalize_weights(self.final_projection), None)


class PolyProjector(Projector):
    def __init__(self, input_features, degree, final_dim):
        super(PolyProjector, self).__init__()

        self.degree = degree
        powers = PolyPowers.calculate(input_features, degree)
        self.register_buffer('powers', powers)

        self.projection = nn.Parameter(
            torch.randn(final_dim, PolyPowers.homopoly(input_features, degree))
        )
        self.reset()
        self.set_training(False)

    def reset(self, force=False):
        if not self.training or force:
            self.projection.data = torch.randn_like(self.projection)

    def forward(self, x):
        x = x.unsqueeze(-1)
        powers = self.powers.unsqueeze(0)
        x = torch.pow(x, powers).prod(dim=1)
        return F.linear(x, normalize_weights(self.projection), None)


def GSWD(x, y, projector, loss_type='l1'):
    projector.reset()
    x_push_forward = projector(x)
    y_push_forward = projector(y)
#     print(x_push_forward.shape, y_push_forward.shape)
    x_sort = torch.sort(x_push_forward, dim=-2)
    y_sort = torch.sort(y_push_forward, dim=-2)

    if loss_type == 'l1':
        return F.l1_loss(x_sort.values, y_sort.values)
    elif loss_type == 'mse':
        return F.mse_loss(x_sort.values, y_sort.values)
    else:
        raise NotImplementedError


print_max = 0


def MGSWD(x, y, projector, loss_type='l1', lr=1e-1, iterations=10):

    x_detach = x.detach()
    y_detach = y.detach()
    global print_max

    projector.set_training(True)
    projector.reset(force=True)
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    for i in range(iterations):
        optimizer.zero_grad()

        loss = -GSWD(
            x_detach, y_detach, projector, loss_type=loss_type,
        )

        loss.backward()

        optimizer.step()
        if print_max % 100 == 0:
            print(f'\tLoss{i}: {loss.item()}')

    print_max += 1
    return GSWD(
        x, y, projector, loss_type=loss_type
    )


class PolyPowers:
    # adapted from https://github.com/kimiandj/gsw/blob/master/code/gsw/gsw.py
    @staticmethod
    def calculate(input_features, degree):
        if input_features == 1:
            return torch.tensor([[degree]])
        else:
            powers = PolyPowers.get_powers(input_features, degree)
            powers = torch.stack([torch.tensor(p) for p in powers], dim=1)
            return powers

    @staticmethod
    def get_powers(dim, degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.
        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]
        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in PolyPowers.get_powers(dim-1, degree-value):
                    yield (value,) + permutation

    @staticmethod
    def homopoly(dim, degree):
        '''
        calculates the number of elements in a homogeneous polynomial
        '''
        return int(
            math.factorial(degree+dim-1) /
            (math.factorial(degree) * math.factorial(dim-1))
        )
