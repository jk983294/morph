import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torchviz import make_dot


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size * seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        return output, hidden


def plot_func(seq_length):
    plt.figure(figsize=(8, 5))

    # generate evenly spaced data pts
    time_steps = np.linspace(0, np.pi, seq_length + 1)
    data = np.sin(time_steps)
    data.resize((seq_length + 1, 1))  # add an input_size dimension

    x = data[:-1]  # all but the last piece of data
    y = data[1:]  # all but the first

    # display the data
    plt.plot(time_steps[1:], x, 'r.', label='input, x')  # x
    plt.plot(time_steps[1:], y, 'b.', label='target, y')  # y

    plt.legend(loc='best')
    plt.show()


def test_dim_correct(model):
    # generate evenly spaced, test data pts
    time_steps = np.linspace(0, np.pi, seq_length)
    data = np.sin(time_steps)
    data.resize((seq_length, 1))
    test_input = torch.Tensor(data).unsqueeze(0)  # (batch, seq_length, output)
    print('Input size: ', test_input.size())

    # test out rnn sizes
    test_out, test_h = model(test_input, None)
    print('Output size: ', test_out.size())
    print('Hidden state size: ', test_h.size())
    make_dot(test_out, params=dict(list(model.named_parameters()))).render("/tmp/rnn_torchviz", format="png")


if __name__ == '__main__':
    # how many time steps/data pts are in one batch of data
    seq_length = 20

    # plot_func(seq_length)

    input_size = 1
    output_size = 1
    hidden_dim = 32
    n_layers = 1
    model = RNN(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)
    print(model)

    # test_dim_correct(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_steps = 75
    print_every = 15

    hidden = None  # initialize the hidden state
    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data
        time_steps = np.linspace(step * np.pi, (step + 1) * np.pi, seq_length + 1)
        data = np.sin(time_steps)
        data.resize((seq_length + 1, 1))  # input_size=1

        x = data[:-1]
        y = data[1:]

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives batch_size dimension
        y_tensor = torch.Tensor(y)

        # outputs from the rnn
        prediction, hidden = model(x_tensor, hidden)

        # Representing Memory
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't back propagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i % print_every == 0:
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.')  # input
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')  # predictions
            plt.show()
