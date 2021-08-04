import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import os


def draw_data(flight_data, x=None, actual_predictions=None):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.xlabel('Months')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(flight_data['passengers'])
    if x is not None and actual_predictions is not None:
        plt.plot(x, actual_predictions)
    plt.show()


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class LstmModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        """
        input_size: the number of features
        self.hidden_h: previous hidden state
        self.hidden_c: previous cell state
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)  # num_layers=1, batch_first=False
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_h = torch.zeros(1, 1, self.hidden_layer_size)
        self.hidden_c = torch.zeros(1, 1, self.hidden_layer_size)

    def reset_hidden_cell(self):
        self.hidden_h = torch.zeros(1, 1, self.hidden_layer_size)
        self.hidden_c = torch.zeros(1, 1, self.hidden_layer_size)

    def forward(self, input_seq):
        """input_seq: (12,) previous 12 month data"""
        input_seq_reshape = input_seq.view(len(input_seq), 1, -1)  # (seq_len, batch, input_size) (12, 1, 1)
        lstm_out, (self.hidden_h, self.hidden_c) = self.lstm(input_seq_reshape, (self.hidden_h, self.hidden_c))
        # lstm_out (seq_len, batch, hidden_size) (12, 1, 100)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))  # (12, 1)
        return predictions[-1]


if __name__ == '__main__':
    flight_data = pd.read_csv(os.path.expanduser('~/github/barn/train/flight.csv'))
    # print(flight_data.head())
    print(flight_data.shape)
    # draw_data(flight_data)

    all_data = flight_data['passengers'].values.astype(float)

    test_data_size = 12
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    print(train_data_normalized[:5])

    train_window = 12  # since we have monthly data
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    print(train_inout_seq[0])

    model = LstmModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)

    """train"""
    epochs = 15
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.reset_hidden_cell()

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    """eval last 12 points"""
    model.eval()
    num_pred = 12
    test_data_normalized = scaler.transform(test_data.reshape(-1, 1)).reshape(-1)
    test_inputs = train_data_normalized[-train_window:].tolist()
    y_hat = []
    for i in range(num_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.reset_hidden_cell()
            y_hat.append(model(seq).item())
            test_inputs.append(test_data_normalized[i])

    actual_predictions = scaler.inverse_transform(np.array(y_hat).reshape(-1, 1))
    # print(actual_predictions)
    x = np.arange(132, 144, 1)
    draw_data(flight_data, x, actual_predictions)
