import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

if __name__ == '__main__':
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Download and load the training data
    trainset = datasets.MNIST('~/junk/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)
    plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
