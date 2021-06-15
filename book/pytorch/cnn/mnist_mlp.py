<<<<<<< HEAD
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from book.pytorch.utils.helper import plot_train_test_loss, get_mnist_loader
import numpy as np


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))  # 随机丢掉一些net
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


if __name__ == '__main__':
    epochs = 5
    batch_size = 64
    model = Classifier()
    criterion: nn.NLLLoss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_loader, test_loader, valid_loader = get_mnist_loader(batch_size=batch_size, valid_size=0.2)

    train_losses, valid_losses = [], []
    valid_loss_min = np.Inf  # set initial "min" to infinity
    for e in range(epochs):
        tot_train_loss = 0

        """train the model"""
        model.train()  # prep model for training
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            tot_train_loss += loss.item() * data.size(0)
        else:
            tot_valid_loss = 0
            valid_correct = 0  # Number of correct predictions

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()  # eval mode 让 dropout ratio = 0, 使用整个net来评估
                for data, target in valid_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    tot_valid_loss += loss.item() * data.size(0)
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == target.view(*top_class.shape)
                    valid_correct += equals.sum().item()

            # Get mean loss to enable comparison between train and test sets
            train_loss = tot_train_loss / len(train_loader.dataset)
            valid_loss = tot_valid_loss / len(test_loader.dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Valid Loss: {:.3f}.. ".format(valid_loss),
                  "Valid Accuracy: {:.3f}".format(valid_correct / len(valid_loader.dataset)))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), '/tmp/model.pt')
                valid_loss_min = valid_loss

    # check out performance
    plot_train_test_loss(train_losses, valid_losses)

    model.load_state_dict(torch.load('/tmp/model.pt'))

    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    test_correct = 0

    model.eval()  # prep model for evaluation
    for data, target in test_loader:
        output = model(data)
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
||||||| 0111d6c
=======
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from book.pytorch.utils.helper import plot_train_test_loss, get_mnist_loader


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))  # 随机丢掉一些net
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


if __name__ == '__main__':
    model = Classifier()
    criterion: nn.NLLLoss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_loader, test_loader = get_mnist_loader()

    epochs = 5
    train_losses, test_losses = [], []
    for e in range(epochs):
        tot_train_loss = 0
        model.train()  # prep model for training
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            tot_train_loss += loss.item() * data.size(0)
        else:
            tot_test_loss = 0
            test_correct = 0  # Number of correct predictions on the test set

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()  # eval mode 让 dropout ratio = 0, 使用整个net来评估
                for data, target in test_loader:
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = model(data)
                    # calculate the loss
                    loss = criterion(output, target)
                    # update test loss
                    tot_test_loss += loss.item() * data.size(0)
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == target.view(*top_class.shape)
                    test_correct += equals.sum().item()

            # Get mean loss to enable comparison between train and test sets
            train_loss = tot_train_loss / len(train_loader.dataset)
            test_loss = tot_test_loss / len(test_loader.dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Test Loss: {:.3f}.. ".format(test_loss),
                  "Test Accuracy: {:.3f}".format(test_correct / len(test_loader.dataset)))

    # check out performance
    plot_train_test_loss(train_losses, test_losses)
>>>>>>> 9d06c5028639fdc99582041dfd5366124fa47f4f
