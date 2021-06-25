import torch
import numpy as np
from book.pytorch.utils.helper import get_mnist_loader
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


class ConvAutoEncoder(nn.Module):
    def __init__(self, encoding_dim):
        super(ConvAutoEncoder, self).__init__()
        # encoder layers
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        self.conv4 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        # encode
        # add hidden layers with relu activation function and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        # decode
        # upsample, followed by a conv layer, with relu activation function
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4(x))
        # upsample again, output should have a sigmoid applied
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv5(x))
        return x


if __name__ == '__main__':
    batch_size = 20
    train_loader, test_loader, valid_loader = get_mnist_loader(batch_size=batch_size, is_norm=False)

    model = ConvAutoEncoder(encoding_dim=32)
    print(model)

    """comparing pixel values in input and output images, it's best to use a loss that meant for a regression task"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data in train_loader:
            images, _ = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # check test
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    output = model(images)
    images = images.numpy()  # prep images for display

    # output is resized into a batch of images
    output = output.view(batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
