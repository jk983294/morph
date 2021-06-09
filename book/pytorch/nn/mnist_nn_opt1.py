import torch
from torch import nn
from book.pytorch.utils import helper
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
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    trainloader, testloader = get_mnist_loader()

    epochs = 10
    train_losses, test_losses = [], []
    for e in range(epochs):
        tot_train_loss = 0
        for images, labels in trainloader:
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_train_loss += loss.item()
        else:
            tot_test_loss = 0
            test_correct = 0  # Number of correct predictions on the test set

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()  # eval mode 让 dropout ratio = 0, 使用整个net来评估
                for images, labels in testloader:
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    tot_test_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_correct += equals.sum().item()
            model.train()  # enable dropout after eval

            # Get mean loss to enable comparison between train and test sets
            train_loss = tot_train_loss / len(trainloader.dataset)
            test_loss = tot_test_loss / len(testloader.dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Test Loss: {:.3f}.. ".format(test_loss),
                  "Test Accuracy: {:.3f}".format(test_correct / len(testloader.dataset)))

    # check out performance
    plot_train_test_loss(train_losses, test_losses)

    # Test out your network!
    model.eval()

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    img = images[0].view(1, 784)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output)

    # Plot the image and probabilities
    helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')
