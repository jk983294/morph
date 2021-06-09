from torch import nn
import torch.nn.functional as F
from book.pytorch.utils import helper
from book.pytorch.utils.helper import get_mnist_loader


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation xW + b
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    model = Network()
    print(model)

    # custom initialization
    model.fc1.bias.data.fill_(0)
    model.fc1.weight.data.normal_(std=0.01)  # sample from random normal

    print(model.fc1.weight)
    print(model.fc1.bias)

    trainloader, testloader = get_mnist_loader()

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)  # [64, 1, 28, 28]
    images.resize_(64, 1, 784)

    # Forward pass through the network
    img_idx = 0
    ps = model.forward(images[img_idx, :])

    img = images[img_idx]
    helper.view_classify(img.view(1, 28, 28), ps)
