import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from book.pytorch.utils.helper import get_mnist_loader
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.leaky_relu(self.fc1(x), 0.2)  # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = self.fc4(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)  # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = torch.tanh(self.fc4(x))
        return out


def real_loss(D_out, smooth=False):
    """
    To help the discriminator generalize better, the labels are reduced a bit from 1.0 to 0.9
    """
    batch_size = D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)  # real labels = 1

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    """https://github.com/udacity/deep-learning-v2-pytorch/blob/master/gan-mnist/MNIST_GAN_Solution.ipynb"""
    batch_size = 64
    train_loader, test_loader, valid_loader = get_mnist_loader(batch_size=batch_size, is_norm=False)

    # Discriminator hyper params
    input_size = 784  # size of input image to discriminator (28*28)
    d_output_size = 1  # Size of discriminator output (real or fake)
    d_hidden_size = 32  # Size of last hidden layer in the discriminator

    # Generator hyper params
    z_size = 100  # Size of latent vector to give to generator
    g_output_size = 784  # Size of discriminator output (generated image)
    g_hidden_size = 32  # Size of first hidden layer in the generator

    D = Discriminator(input_size, d_hidden_size, d_output_size)
    G = Generator(z_size, g_hidden_size, g_output_size)
    print(D)
    print(G)

    # Optimizers
    lr = 0.002
    d_optimizer = optim.Adam(D.parameters(), lr)
    g_optimizer = optim.Adam(G.parameters(), lr)

    num_epochs = 100
    print_every = 400
    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # sampling fixed data throughout training, it allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images * 2 - 1  # rescale input images from [0,1) to [-1, 1)

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
            d_optimizer.zero_grad()

            # 1. Train with real images, Compute the discriminator losses on real images smooth the real labels
            D_real = D(real_images)
            d_real_loss = real_loss(D_real, smooth=True)

            # 2. Train with fake images, Generate fake images, gradients don't have to flow during this step
            with torch.no_grad():
                z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                z = torch.from_numpy(z).float()
                fake_images = G(z)

            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)  # Generate fake images

            # Compute the discriminator losses on fake images using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake)  # use real loss to flip labels

            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

        # AFTER EACH EPOCH
        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))

        # generate and save sample, fake images
        G.eval()  # eval mode for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to train mode

    # Save training generator samples
    with open('/tmp/train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()

    # showing the generated images as the network was training, every 10 epochs
    rows = 10
    cols = 6
    fig, axes = plt.subplots(figsize=(7, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)
    for sample, ax_row in zip(samples[::int(len(samples) / rows)], axes):
        for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
            img = img.detach()
            ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    plt.show()

    # randomly generated, new latent vectors
    sample_size = 16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    rand_z = torch.from_numpy(rand_z).float()
    G.eval()  # eval mode
    rand_images = G(rand_z)
    view_samples(0, [rand_images])
