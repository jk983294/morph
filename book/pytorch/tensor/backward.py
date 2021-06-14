import torch
import numpy as np


def hook_demo():
    v = torch.tensor([0., 0., 0.], requires_grad=True)
    h = v.register_hook(lambda grad: grad * 2)  # double the gradient
    v.backward(torch.tensor([1., 2., 3.]))
    print(v.grad)
    h.remove()  # removes the hook


if __name__ == '__main__':
    x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
    out = x.pow(2).sum()
    out.backward()
    print(out)
    print(x.grad)

    hook_demo()
