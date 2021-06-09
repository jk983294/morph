import torch
from torch import nn
from book.pytorch.utils import helper
from torch import optim

from book.pytorch.utils.helper import get_mnist_loader

if __name__ == '__main__':
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    trainloader, testloader = get_mnist_loader()

    epochs = 5
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / len(trainloader)}")

    # check out performance
    images, labels = next(iter(trainloader))

    img = images[0].view(1, 784)
    with torch.no_grad():  # Turn off gradients to speed up this part
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    helper.view_classify(img.view(1, 28, 28), ps)
