import torch
import numpy as np
from book.pytorch.utils.helper import get_mnist_loader
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


class ConvDenoiser(nn.Module):
    def __init__(self, encoding_dim):
        super(ConvDenoiser, self).__init__()
        # encoder layers
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        # encode
        # add hidden layers with relu activation function and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # decode
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = torch.sigmoid(self.conv_out(x))
        return x


if __name__ == '__main__':
    """
    used to denoise images quite successfully just by training the network on noisy images
    """
    batch_size = 20
    train_loader, test_loader, valid_loader = get_mnist_loader(batch_size=batch_size, is_norm=False)

    model = ConvDenoiser(encoding_dim=32)
    print(model)

    """comparing pixel values in input and output images, it's best to use a loss that meant for a regression task"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    noise_factor = 0.5  # for adding noise to images

    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data in train_loader:
            images, _ = data

            # add random noise to the input images
            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            # the "target" is still the original, not-noisy images
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

    # add noise to the test images
    noisy_imgs = images + noise_factor * torch.randn(*images.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)

    output = model(noisy_imgs)
    noisy_imgs = noisy_imgs.numpy()  # prep images for display

    # output is resized into a batch of images
    output = output.view(batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for noisy_imgs, row in zip([noisy_imgs, output], axes):
        for img, ax in zip(noisy_imgs, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
