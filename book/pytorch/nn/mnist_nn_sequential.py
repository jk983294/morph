from collections import OrderedDict
from torch import nn
from book.pytorch.utils import helper
from book.pytorch.utils.helper import get_mnist_loader


def test_idx():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.Softmax(dim=1))
    print(model)

    print(model[0].weight)
    print(model[0].bias)

    trainloader, testloader = get_mnist_loader()

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)  # [64, 1, 28, 28]
    images.resize_(64, 1, 784)

    # Forward pass through the network
    ps = model.forward(images[0, :])
    helper.view_classify(images[0].view(1, 28, 28), ps)


def test_named_dict():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('output', nn.Linear(hidden_sizes[1], output_size)),
        ('softmax', nn.Softmax(dim=1))]))

    print(model)
    print(model[0])
    print(model.fc1)

    trainloader, testloader = get_mnist_loader()

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)  # [64, 1, 28, 28]
    images.resize_(64, 1, 784)

    # Forward pass through the network
    ps = model.forward(images[0, :])
    helper.view_classify(images[0].view(1, 28, 28), ps)


if __name__ == '__main__':
    # test_idx()
    test_named_dict()
