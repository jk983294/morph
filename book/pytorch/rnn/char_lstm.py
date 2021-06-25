import os
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torchviz import make_dot

from book.pytorch.utils.train_helper import one_hot_encode


class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001, train_on_gpu=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.train_on_gpu = train_on_gpu

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # create two new tensors with sizes n_layers * batch_size * n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


def get_batches(text_, batch_size, seq_length):
    """
    Create a generator that returns batches of size batch_size x seq_length from arr.

    Arguments:
    text_: data you want to make batches from
    N batch_size: Batch size, the number of sequences per batch
    M seq_length: Number of encoded chars in a sequence
    K n_batches: total number of batches
    """

    batch_size_total = batch_size * seq_length
    n_batches = len(text_) // batch_size_total

    # Keep only enough characters to make full batches
    text_ = text_[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    text_ = text_.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, text_.shape[1], seq_length):
        # The features
        x = text_[:, n:n + seq_length]  # N * M, batch_size * seq_length
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], text_[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], text_[:, 0]
        yield x, y


def test_batch(encoded_):
    batches = get_batches(encoded_, batch_size=4, seq_length=10)
    x, y = next(batches)
    print(x)
    print(y)


if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU!')
    else:
        print('No GPU available, training on CPU; consider making n_epochs very small.')

    with open(os.path.expanduser('~/github/barn/deep/anna.txt'), 'r') as f:
        text = f.read()

    # encode the text and map each character to an integer and vice versa
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    data = np.array([char2int[ch] for ch in text])  # text to int

    # print(text[:100])
    # print(data[:100])
    # test_batch(data)

    epochs = 10
    batch_size = 128  # Number of mini-sequences per mini-batch, aka batch size
    seq_length = 100  # Number of character steps per mini-batch
    clip = 5  # gradient clipping
    valid_fraction = 0.1  # Fraction of data to hold out for validation
    print_every = 10
    n_hidden = 512
    n_layers = 2

    net = CharRNN(chars, n_hidden, n_layers, train_on_gpu=train_on_gpu)
    if train_on_gpu:
        net.cuda()
    print(net)

    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data) * (1 - valid_fraction))
    data, val_data = data[:val_idx], data[val_idx:]

    net.train()  # prepare training
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length).long())
                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after validation
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
