import torch
from torch import nn
from torchvision import datasets, transforms


def enable_auto_grad():
    x = torch.zeros(1, requires_grad=True)
    with torch.no_grad():
        y = x * 2
    print(y.requires_grad)

    torch.set_grad_enabled(False)
    torch.set_grad_enabled(True)


if __name__ == '__main__':
    enable_auto_grad()

    x = torch.randn(2, 2, requires_grad=True)
    y = x ** 2
    print(y.grad_fn)  # shows the function that generated this variable
    z = y.mean()
    z.backward()
    print(x.grad)
