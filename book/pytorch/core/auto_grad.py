<<<<<<< HEAD
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


def jacobian_demo():
    """y=(x+1)^2, y_prime = 2 * (x + 1)"""
    inp = torch.eye(5, requires_grad=True)
    out = (inp + 1).pow(2)
    out.backward(torch.ones_like(inp), retain_graph=True)
    print("First call\n", inp.grad)
    out.backward(torch.ones_like(inp), retain_graph=True)  # 累计梯度
    print("\nSecond call\n", inp.grad)
    inp.grad.zero_()
    out.backward(torch.ones_like(inp), retain_graph=True)
    print("\nCall after zeroing gradients\n", inp.grad)


if __name__ == '__main__':
    enable_auto_grad()

    x = torch.randn(2, 2, requires_grad=True)
    y = x ** 2
    print(y.grad_fn)  # shows the function that generated this variable
    z = y.mean()
    z.backward()
    print(x.grad)

    """https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"""
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print('Gradient function for z =', z.grad_fn)
    print('Gradient function for loss =', loss.grad_fn)
    loss.backward()
    print(w.grad)
    print(b.grad)

    jacobian_demo()
||||||| 0111d6c
=======
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
>>>>>>> 9d06c5028639fdc99582041dfd5366124fa47f4f
