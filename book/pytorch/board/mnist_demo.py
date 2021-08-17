import copy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from book.pytorch.utils.helper import plot_train_test_loss, get_mnist_loader
import numpy as np
from tensorboardX import SummaryWriter


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


if __name__ == '__main__':
    writer = SummaryWriter('runs')
    epochs = 5
    batch_size = 64
    model = Classifier()
    criterion: nn.NLLLoss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_loader, test_loader, valid_loader = get_mnist_loader(batch_size=batch_size, valid_size=0.2)

    best_model = None
    train_losses, valid_losses = [], []
    valid_loss_min = np.Inf  # set initial "min" to infinity
    for e in range(epochs):
        tot_train_loss = 0

        """train the model"""
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tot_train_loss += loss.item() * data.size(0)
        else:
            tot_valid_loss = 0
            valid_correct = 0

            with torch.no_grad():
                model.eval()
                for data, target in valid_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    tot_valid_loss += loss.item() * data.size(0)
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == target.view(*top_class.shape)
                    valid_correct += equals.sum().item()

            # Get mean loss to enable comparison between train and test sets
            train_loss = tot_train_loss / len(train_loader.sampler)
            valid_loss = tot_valid_loss / len(valid_loader.sampler)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            writer.add_scalar('Train/Loss', train_loss, e)
            writer.add_scalar('Valid/Loss', valid_loss, e)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Valid Loss: {:.3f}.. ".format(valid_loss),
                  "Valid Accuracy: {:.3f}".format(valid_correct / len(valid_loader.sampler)))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))
                best_model = copy.deepcopy(model)
                valid_loss_min = valid_loss

    # check out performance
    plot_train_test_loss(train_losses, valid_losses)

    model = best_model

    test_loss = 0.0
    test_correct = 0

    best_model.eval()
    for data, target in test_loader:
        output = best_model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == target.view(*top_class.shape)
        test_correct += equals.sum().item()

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print("Test Loss: {:.3f}.. ".format(test_loss),
          "Test Accuracy: {:.3f}".format(test_correct / len(test_loader.dataset)))
