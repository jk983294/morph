import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
import os

from book.pytorch.ts.flight_no_batch import draw_data


def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)


class LstmModel(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (120, 12, 1) (batch, seq_len, output_size)
        h_0: (1, 120, 2) (num_layers * output_size, batch, hidden_size)
        """
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        lstm_out, (h_out, _) = self.lstm(x, (h_0, c_0))  # lstm_out (120, 12, 2)
        h_out = h_out.view(-1, self.hidden_size)  # (120, 2)
        y_hat = self.linear(h_out)  # (120, 1)
        return y_hat


if __name__ == '__main__':
    flight_data = pd.read_csv(os.path.expanduser('~/github/barn/train/flight.csv'))
    # print(flight_data.head())
    print(flight_data.shape)
    # draw_data(flight_data)

    all_data = flight_data['passengers'].values.astype(float)

    test_data_size = 12
    seq_length = 12
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    x, y = sliding_windows(train_data_normalized, seq_length)
    train_X = Variable(torch.Tensor(np.array(x)))  # (120, 12, 1)
    train_y = Variable(torch.Tensor(np.array(y)))  # (120, 1)

    input_size = 1
    hidden_size = 2
    num_layers = 1
    output_size = 1
    model = LstmModel(output_size, input_size, hidden_size, num_layers)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)

    """train"""
    epochs = 2500
    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model(train_X)

        single_loss = loss_function(y_pred, train_y)
        single_loss.backward()
        optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    """eval last 12 points"""
    model.eval()
    num_pred = 12
    test_data_normalized = scaler.transform(test_data.reshape(-1, 1)).reshape(-1)
    test_inputs = np.concatenate((train_data_normalized[-num_pred:].reshape(-1), test_data_normalized))
    x_test, y_test = sliding_windows(test_inputs.reshape(-1, 1), seq_length)
    test_X = Variable(torch.Tensor(np.array(x_test)))  # (12, 12, 1)
    with torch.no_grad():
        y_hat = model(test_X)

    actual_predictions = scaler.inverse_transform(np.array(y_hat).reshape(-1, 1))
    # print(actual_predictions)
    x = np.arange(132, 144, 1)
    draw_data(flight_data, x, actual_predictions)
