import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def normalize_weights(weights):
    return weights / torch.norm(weights, dim=1, keepdim=True)


class Projector(nn.Module):
    """
        Base Class for linear and non-linear projections
    """
    def __init__(self):
        super(Projector, self).__init__()
        self.training = False

    def set_training(self, status):
        """
        Set if params are frozen or not
        """
        self.training = status
        for param in self.parameters():
            param.requires_grad = self.training

    def reset(self, force=False):
        return NotImplemented


class LinearProjector(Projector):
    """
    A Linear Projector for Sliced Wasserstein Distance
    """
    def __init__(self, input_features, final_dim):
        """
        Init for LinearProjector
        :param input_features: channel dimension of input features to be projected
        :param final_dim: the final output dimension of projected features
        """
        super(LinearProjector, self).__init__()
        self.projection = nn.Parameter(torch.randn(final_dim, input_features))
        self.reset()
        self.set_training(False)

    def reset(self, force=False):
        """
        Reset projection
        :param force: Whether or not to force reset even if training
        """
        if not self.training or force:
            self.projection.data = torch.randn_like(self.projection)

    def forward(self, x):
        return F.linear(x, normalize_weights(self.projection), None)


class NNProjector(Projector):
    """
    A Neural Network Projector for Sliced Wasserstein Distance
    """
    def __init__(
        self, num_layer, input_features, hidden_dim, final_dim,
    ):
        """
        Init for NNProjector
        :param num_layer: Number of layers of neural network
        :param input_features: channel dimension of input features to be projected
        :param hidden_dim: dimension of hidden layers
        :param final_dim: the final output dimension of projected features
        """
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
        """
        Reset projection
        :param force: Whether or not to force reset even if training
        """
        if not self.training or force:
            for module in self.projection.children():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(
                        module.weight,
                        gain=nn.init.calculate_gain('leaky_relu')
                    )
            self.final_projection.data = torch.randn_like(
                self.final_projection
            )

    def forward(self, x):
        x = self.projection(x)
        return F.linear(x, normalize_weights(self.final_projection), None)


class PolyProjector(Projector):
    """
    A Homogeneous Polynomial Projector for Sliced Wasserstein Distance
    """
    def __init__(self, input_features, degree, final_dim):
        """
        Init for PolyProjector
        :param input_features: channel dimension of input features to be projected
        :param degree: degree of polynomial to use, eg 3, 5
        :param final_dim: the final output dimension of projected features
        """
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
        """
        Reset projection
        :param force: Whether or not to force reset even if training
        """
        if not self.training or force:
            self.projection.data = torch.randn_like(self.projection)

    def forward(self, x):
        x = x.unsqueeze(-1)
        powers = self.powers.unsqueeze(0)
        x = torch.pow(x, powers).prod(dim=1)
        return F.linear(x, normalize_weights(self.projection), None)


class GSWD(nn.Module):
    """
    A Class to calculate the Generalized Sliced Wasserstein Distance
    """
    def __init__(self, projector, loss_type='l1'):
        """
        :param projector: A `Projector` instance
        :param loss_type: mse or l1
        """
        super(GSWD, self).__init__()
        assert loss_type in ('l1', 'mse')
        self.loss = getattr(torch.nn.functional, f'{loss_type}_loss')
        self.projector = projector

    def forward(self, x, y):
        self.projector.reset()
        x_push_forward = self.projector(x)
        y_push_forward = self.projector(y)

        x_sort = torch.sort(x_push_forward, dim=-2)
        y_sort = torch.sort(y_push_forward, dim=-2)

        return self.loss(x_sort.values, y_sort.values)


class MGSWD(GSWD):
    """
    A Class to calculate the Max Generalized Sliced Wasserstein Distance
    """
    def __init__(
        self, projector, loss_type, lr=1e-1, iterations=10, verbose=False
    ):
        """
        :param projector: A `Projector` instance
        :param loss_type: mse or l1
        :param lr: the learning rate for sub optimzation
        :param iterations: the number of iterations to run
        :param verbose: Whether or not to print sub optimization losses
        """
        super(MGSWD, self).__init__(projector, loss_type)
        self.lr = lr
        self.iterations = iterations
        self.print_max = 0
        self.verbose = verbose

    def forward(self, x, y):
        x_detach = x.detach()
        y_detach = y.detach()

        with torch.enable_grad():
            self.projector.set_training(True)
            self.projector.reset(force=True)
            optimizer = torch.optim.Adam(self.projector.parameters(), lr=self.lr)
            for i in range(self.iterations):
                optimizer.zero_grad()

                loss = -super(MGSWD, self).forward(x_detach, y_detach)

                loss.backward()

                optimizer.step()
                if self.verbose and self.print_max % 100 == 0:
                    print(f'\tLoss{i}: {loss.item()}')

        self.print_max += 1
        return super(MGSWD, self).forward(x, y)


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
