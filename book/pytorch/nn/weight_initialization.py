import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from book.pytorch.utils.helper import get_mnist_loader
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, hidden_1=256, hidden_2=128, constant_weight=None):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.dropout = nn.Dropout(0.2)

        # initialize the weights to a specified, constant value
        if constant_weight is not None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def _get_loss_acc(model, train_loader, valid_loader):
    """
    Get losses and validation accuracy of example neural network
    """
    n_epochs = 2
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_batch = []

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())

    # after training for 2 epochs, check validation accuracy
    correct = 0
    total = 0
    for data, target in valid_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()

    valid_acc = correct.item() / total
    return loss_batch, valid_acc


def compare_init_weights(model_list, plot_title, train_loader, valid_loader, plot_n_batches=100):
    """
    Plot loss and print stats of weights using an example neural network
    """
    colors = ['r', 'b', 'g', 'c', 'y', 'k']
    label_accs = []
    label_loss = []

    assert len(model_list) <= len(colors), 'Too many initial weights to plot'

    for i, (model, label) in enumerate(model_list):
        loss, val_acc = _get_loss_acc(model, train_loader, valid_loader)

        plt.plot(loss[:plot_n_batches], colors[i], label=label)
        label_accs.append((label, val_acc))
        label_loss.append((label, loss[-1]))

    plt.title(plot_title)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    print('After 2 Epochs:')
    print('Validation Accuracy')
    for label, val_acc in label_accs:
        print('  {:7.3f}% -- {}'.format(val_acc * 100, label))
    print('Training Loss')
    for label, loss in label_loss:
        print('  {:7.3f}  -- {}'.format(loss, label))


def hist_dist(title, distribution_tensor, hist_range=(-4, 4)):
    """
    Display histogram of values in a given distribution tensor
    """
    plt.title(title)
    plt.hist(distribution_tensor, np.linspace(*hist_range, num=len(distribution_tensor) // 2))
    plt.show()


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


def weights_init_uniform_center(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a centered, uniform distribution to the weights
        m.weight.data.uniform_(-0.5, 0.5)
        m.bias.data.fill_(0)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    num_workers = 0
    batch_size = 64
    valid_size = 0.2
    train_loader, test_loader, valid_loader = get_mnist_loader(batch_size, valid_size, num_workers)

    # model_0 = Net(constant_weight=0)
    # model_1 = Net(constant_weight=1)
    # model_list = [(model_0, 'All Zeros'), (model_1, 'All Ones')]
    # compare_init_weights(model_list, 'All Zeros vs All Ones', train_loader, valid_loader)

    model_uniform = Net()
    model_uniform.apply(weights_init_uniform)
    compare_init_weights([(model_uniform, 'Uniform Weights')], 'Uniform', train_loader, valid_loader)

    """ Good practice is to start your weights in the range of [-y, y] where y = 1 / sqrt(n) """
    model_centered = Net()
    model_centered.apply(weights_init_uniform_center)
    model_rule = Net()
    model_rule.apply(weights_init_uniform_rule)

    model_list = [(model_centered, 'Centered Weights [-0.5, 0.5)'), (model_rule, 'General Rule [-y, y)')]
    compare_init_weights(model_list, '[-0.5, 0.5) vs [-y, y)', train_loader, valid_loader)
