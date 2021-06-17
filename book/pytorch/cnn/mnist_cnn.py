import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from book.pytorch.utils.helper import plot_train_test_loss
import numpy as np
from torchvision import datasets, transforms


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], 64 * 4 * 4)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    batch_size = 64
    valid_size = 0.2
    num_workers = 4
    model = Classifier()
    criterion: nn.NLLLoss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    data_dir = '~/junk/Cat_Dog_data/'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(32),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])
    test_transforms = transforms.Compose([transforms.Resize(36),
                                          transforms.CenterCrop(32),
                                          transforms.ToTensor()])
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    if train_on_gpu:
        model.cuda()

    epochs = 50
    train_losses, valid_losses = [], []
    valid_loss_min = np.Inf  # set initial "min" to infinity
    for e in range(epochs):
        tot_train_loss = 0

        """train the model"""
        model.train()  # prep model for training
        for data, target in train_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tot_train_loss += loss.item() * data.size(0)
        else:
            tot_valid_loss = 0
            valid_correct = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()  # eval mode 让 dropout ratio = 0, 使用整个net来评估
                for data, target in valid_loader:
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()
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

            # At completion of epoch
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Valid Loss: {:.3f}.. ".format(valid_loss),
                  "Valid Accuracy: {:.3f}".format(valid_correct / len(valid_loader.sampler)))

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
