from torchvision import datasets
from torchvision import transforms
import torch

default_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))])

def load_mnist_dataset(path, transform = default_transform, batch_size = 64):
    # MNIST dataset
    dataset = datasets.MNIST(root='../../Data',
                   train=True,
                   transform=transform,
                   download=True)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
    return data_loader
