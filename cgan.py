import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torchvision.utils as tutils
import imageio
from PIL import Image
import pickle
from random import randint
#
from utils import *
import models
import data_loader
import train

path = './data'
list_lrates = [0.1, 0.01, 0.001, 0.0001]
list_optimizers = ['Adam', 'SGD', 'Adadelta', 'RMSprop']
batch_size = 64
dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!

# Helper routines
if torch.cuda.is_available():
    IS_CUDA = True

num_gens = len(list_lrates) * len(list_optimizers)
data_loader = data_loader.load_mnist_dataset(path)

generators = []
optimizers = []
config     = []
discriminator = models.Discriminator.type(dtype)
#discriminator.apply(models.initialize_weights)
discriminator = discriminator.cuda(0)
lossManager = LossModule(numberOfGens = num_gens)
lossCriterion = nn.BCELoss()
D_opt = torch.optim.Adam(discriminator.parameters(), lr = 0.0001)

# Make the generators
for lr in list_lrates:
    for opt in list_optimizers:
        t_gen = models.Generator().type(dtype)
        t_gen = t_gen.cuda(0)
        # t_gen.apply(models.initialize_weights)
        generators.append(t_gen)
        optimizers.append(get_optimizer(opt, t_gen, lr))
        config.append([lr, opt])

BestPerformingGenerator = randint(0, num_gens-1)
BestPerformingGenerator = train.train(data_loader, discriminator, D_opt, BestPerformingGenerator, generators, optimizers, config, lossManager, lossCriterion, batch_size)
