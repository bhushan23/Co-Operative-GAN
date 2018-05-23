import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as tutils
import imageio
from PIL import Image
import pickle

from utils import *
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!

def train(data_loader, Discriminator, D_opt, BestPerformingGenerator, Generators, G_Optimizers, config, lossManager, lossCriterion, batch_size, Generator_input = 100, num_epochs = 10, d_iter = 1):
    if torch.cuda.is_available():
        IS_CUDA = True
    NumberOfGenerators = len(Generators)

    for epoch in range(num_epochs):
        lossList = [0.0] * NumberOfGenerators
        for data in data_loader:
            image, _  = data
            image = var(image.view(image.size(0),  -1))

            # Train Discriminator
            # for k in range(0, d_iter):
            # for each in Generators:
            D_real = Discriminator(image)
            # For Log(1 - D(G(Z)))
            Z_noise = var(torch.randn(batch_size, 100))
            #print Z_noise.shape
            #print type(Gen)
            G_fake = Generators[BestPerformingGenerator](Z_noise) #each(Z_noise)
            #print G_fake.shape
            D_fake = Discriminator(G_fake)

            # Calculate Discriminator Loss
            D_real_loss = lossCriterion(D_real, var(torch.ones(batch_size, 1)))
            D_fake_loss = lossCriterion(D_fake, var(torch.zeros(batch_size, 1)))
            D_loss = D_real_loss + D_fake_loss

            # Backprop Discriminator
            Discriminator.zero_grad()
            D_loss.backward()
            D_opt.step()
            # print 'Discriminator Loop for: {}: {}'.format(i, D_loss.data[0])

            # Find best performing Generator
            i = 0
            GeneratorLoss = []
            for each, each_opt in zip(Generators, G_Optimizers):
                # print('Training Gen:', i)
                Z_noise = var(torch.randn(batch_size, Generator_input))
                G_fake = each(Z_noise)
                #print G_fake1.shape
                #print type(each)
                D_fake = Discriminator(G_fake)
                # Compute Generator Loss
                G_loss = lossCriterion(D_fake, var(torch.ones(batch_size, 1)))
                GeneratorLoss.append(G_loss)
                lossList[i] += (float(G_loss.data[0]))
                i = i + 1
                Discriminator.zero_grad()
                each.zero_grad()
                G_loss.backward()
                each_opt.step()

        BestPerformingGenerator = lossList.index(min(i for i in lossList if i is not 0)) # lossList.index(min(lossList)) # earlier was min
        print(lossList)
        for i in range(0, NumberOfGenerators):
            if i != BestPerformingGenerator:
                prev = Generators[i]
                Generators[i] = Generator()
                if IS_CUDA:
                    Generators[i].cuda()
                Generators[i].load_state_dict(Generators[BestPerformingGenerator].state_dict())
                G_Optimizers[i] =  get_optimizer(config[i][1], Generators[i], config[i][0])
                # torch.optim.Adam(Generators[i].parameters(), lr = 0.0001)
                # G_Optimizers[i].load_state_dict(G_Optimizers[BestPerformingGenerator].state_dict())
                if Generators[i] == prev:
                    print('ERROR: Generator unchanged')


        print('{} Epoch [{}/{}], Discriminator {:.4f}, Best Generator[{}] {:.4f}'.format(BestPerformingGenerator, epoch+1, num_epochs, D_loss.data[0], BestPerformingGenerator, GeneratorLoss[BestPerformingGenerator].data[0]))
        lossManager.insertDiscriminatorLoss(D_loss.data[0])
        lossManager.insertGeneratorLoss(G_loss.data[0])
        # lossManager.insertGeneratorList(GeneratorLoss)
        pic = Generators[BestPerformingGenerator](fixed_x)
        pic = denorm(pic.data)
        outputImages.append(pic)
        #torchvision.utils.save_image(pic, path+'image_{}.png'.format(epoch))
        save_image(pic, path+'image_{}.png'.format(epoch))
    return BestPerformingGenerator
