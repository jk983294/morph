import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import _ContextMethodMixin
from torchvision import datasets, transforms


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        # forward有几个入参，就有几个gradient
        return grad_input, grad_weight, grad_bias


class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        ctx.set_materialize_grads(False)
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        """
        y = 2*x, z = 3 * y, z = 6 * x
        ∂z/∂y = 3 * ∂z/∂z, grad_output = ∂z/∂z = 1
        ∂z/∂x = ∂y/∂x * ∂z/∂y, grad_output = ∂z/∂y = 3
        Gradients of non-Tensor arguments to forward must be None
        """
        if grad_output is None:
            return None, None  # skip unnecessary computations

        return grad_output * ctx.constant, None


if __name__ == '__main__':
    input_ = torch.rand([1, 3])
    ctx = _ContextMethodMixin()
    print(input_)
    o_ = MulConstant.forward(ctx, input_, 3)
    print('forward', o_)

    grad_ = MulConstant.backward(ctx, torch.ones([1, 3]))
    print('backward', grad_)
