import torch.nn as nn
import torch.nn.functional as F
import torch
batch_size = 64
dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
###  Model for MNIST

def build_dc_classifier():#mnist
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 36, 4, stride = 1),
        nn.LeakyReLU(0.02),
        nn.MaxPool2d(2, stride = 2),
        nn.Conv2d(36, 72, 4, stride = 1),
        nn.LeakyReLU(0.02),
        nn.MaxPool2d(2, stride = 2),
        Flatten(),
        nn.Linear(1152, 1024),
        nn.LeakyReLU(0.02),
        nn.Linear(1024, 1),
        nn.Sigmoid()
    )


def build_dc_generator(noise_dim=100):#mnist
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return nn.Sequential(
        nn.Linear(noise_dim, 1500),
        nn.ReLU(),
        nn.BatchNorm1d(1500),
        nn.Linear(1500, 5880),
        nn.BatchNorm1d(5880),
        Unflatten(batch_size, 120, 7, 7),
        nn.ConvTranspose2d(120, 60, 4, stride = 2, padding = 1),
        nn.ReLU(),
        nn.BatchNorm2d(60),
        nn.ConvTranspose2d(60, 1, 4, stride = 2, padding = 1),
        nn.Tanh(),
        Flatten()
    )


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        x = x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
        return x

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        x = x.view(self.N, self.C, self.H, self.W)
        return x

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)


