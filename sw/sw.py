import torch
import torch.nn as nn


class GRT(nn.Module):
    def reset_parameters(self):
        raise NotImplementedError

    def set_theta(self, theta):
        raise NotImplementedError

    # Adapted from https://github.com/kimiandj/gsw/blob/master/code/gsw/gsw.py
    def gsw(self, x, y, theta='reset', loss=torch.nn.functional.mse_loss):
        """
        theta can be one of None, 'reset', tensor, or List[tensor]
        """
        assert x.shape == y.shape

        if theta == 'reset':
            self.reset_parameters()
        elif theta is None:
            pass
        else:
            # assume theta is tensor or List[Tensor]
            self.set_theta(theta)

        x_slice = self(x)
        y_slice = self(y)

        x_sorted = torch.sort(x_slice, dim=-1).values
        y_sorted = torch.sort(y_slice, dim=-1).values

        return loss(x_sorted, y_sorted)


class LinearGRT(GRT):
    def __init__(self, in_features, directions, requires_grad=False):
        super(LinearGRT, self).__init__()

        self.theta = nn.Parameter(torch.randn(in_features, directions))
        self.theta.requires_grad_(requires_grad)

    def reset_parameters(self):
        self.theta.data = torch.randn_like(self.theta.data)

    def set_theta(self, theta):
        assert self.theta.data.shape == theta.shape
        device = self.theta.device
        self.theta.data = theta
        self.theta = self.theta.to(device)

    def forward(self, x, projection_dim=-1):
        # assumes x is laid out as B,..., projection_dim, ...sequence_dims
        if projection_dim != -1:
            # need to permute project dim to end
            indices = list(range(len(x.shape)))
            indices = (
                indices[:projection_dim] + indices[projection_dim+1:]
                + [indices[projection_dim]]
            )
            x = x.rmute(*indices)

        # norm directions
        directions = self.theta / torch.norm(self.theta, dim=0, keepdim=True)
        x = x @ directions
        if projection_dim != -1:
            # need to permute back to original shape
            indices = list(range(len(x.shape)))
            indices = (
                indices[:projection_dim]
                + [indices[-1]]
                + indices[projection_dim:-1]
            )
            x = x.permute(*indices)
        return x


class CircularGRT(GRT):
    def __init__(self, requires_grad=False):
        super(LinearGRT, self).__init__()

    def reset_parameters(self):
        self.theta.data = torch.randn_like(self.theta.data)

    def set_theta(self, theta):
        assert self.theta.data.shape == theta.shape
        device = self.theta.device
        self.theta.data = theta
        self.theta = self.theta.to(device)

    def forward(self, x, projection_dim=-1):
        pass

class PolyGRT(GRT):
    def __init__(self, requires_grad=False):
        super(LinearGRT, self).__init__()

    def reset_parameters(self):
        pass

    def set_theta(self, theta):
        pass

    def forward(self, x):
        pass

class NNGRT(GRT):
    def __init__(self, requires_grad=False):
        super(LinearGRT, self).__init__()

    def reset_parameters(self):
        pass

    def set_theta(self, theta):
        pass

    def forward(self, x):
        pass

