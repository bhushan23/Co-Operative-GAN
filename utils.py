import torch
from torchvision import datasets
from torchvision import transforms
import torchvision
import matplotlib
from torch.autograd import Variable
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torchvision.utils as tutils
import imageio
from PIL import Image


# Helper routines
if torch.cuda.is_available():
    IS_CUDA = True

def var(x):
    if torch.cuda.is_available():
        x = x.cuda(0)
    return Variable(x)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def get_optimizer(name, model, learning_rate):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr = learning_rate)
    elif name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr = learning_rate)
    elif name == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr = learning_rate)
    elif name == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr = learning_rate)
    else:
        print('ERROR: {} not defined'.format(name))
        return None

def generate_animation(root, epoch, name):
    images = []
    for e in range(epoch):
        img_name = root+'/image_'+str(e)+'.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root+ '/' + name +'.gif', images, fps=5)

def save_image(pic, path):
    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

class LossModule:
    def __init__(self, numberOfGens = 1):
        self.D_loss = []
        self.G_loss = []
        self.Many_G_loss = [[]] * (numberOfGens+1)

    def insertDiscriminatorLoss(self, lossVal):
        self.D_loss.append(lossVal)

    def insertGeneratorLoss(self, lossVal):
        self.G_loss.append(lossVal)

    def insertGeneratorList(self, lossList):
        for i in range(0, len(lossList)):
            self.Many_G_loss[i].append(lossList[i].data[0])

    def getDiscriminatorLoss(self):
        return self.D_loss

    def getGeneratorLoss(self):
        return self.G_loss

    def getGeneratorsList(self):
        return self.Many_G_loss

    def drawLossPlot(self, showPlot = False, savePlot = True, label = "Vanilla GAN Loss", genList = False):
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(label)
        if genList == True:
            i = 0
            for each in self.Many_G_loss:
                i = i + 1
                plt.plot(self.G_loss, label='Generator Loss {}'.format(i))
        else:
            plt.plot(self.G_loss, label='Generator Loss')
        plt.plot(self.D_loss, label='Discriminator Loss')
        legend = plt.legend(loc='upper right', shadow=True)

        if showPlot:
            plt.show()
        if savePlot:
            plt.savefig(label+'Loss_Plot_Vanilla_GAN_'+str(num_epochs)+'.png')
