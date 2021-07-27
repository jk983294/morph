import torch
import torch.nn as nn
from torch import optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))


if __name__ == '__main__':
    """
    it is useful when your model is too large for a single GPU.
    model parallel is to place different sub-networks of a model onto different devices, 
    and implement the forward method accordingly to move intermediate outputs across devices
    """
    model = ToyModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to('cuda:1')
    loss_fn(outputs, labels).backward()
    optimizer.step()
