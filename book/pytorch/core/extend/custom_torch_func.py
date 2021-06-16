import torch
import functools

HANDLED_FUNCTIONS = {}


class DiagonalTensor(object):
    def __init__(self, N, value):
        self._N = N
        self._value = value

    def __repr__(self):
        return "DiagonalTensor(N={}, value={})".format(self._N, self._value)

    def tensor(self):
        return self._value * torch.eye(self._N)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, DiagonalTensor)) for t in types):
            args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
            return func(*args, **kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(torch_function):
    """Register a torch function override for DiagonalTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.mean)
def mean(input_):
    return float(input_._value) / input_._N


def ensure_tensor(data):
    if isinstance(data, DiagonalTensor):
        return data.tensor()
    return torch.as_tensor(data)


@implements(torch.add)
def add(input_, other):
    try:
        if input_._N == other._N:
            return DiagonalTensor(input_._N, input_._value + other._value)
        else:
            raise ValueError("Shape mismatch!")
    except AttributeError:
        return torch.add(ensure_tensor(input_), ensure_tensor(other))


if __name__ == '__main__':
    d = DiagonalTensor(5, 2)
    print(d)
    print(d.tensor())
    print(torch.mean(d))  # torch.mean 特化来处理 DiagonalTensor

    s = DiagonalTensor(2, 2)
    print(torch.add(s, s))
    t = torch.tensor([[1, 1, ], [1, 1]])
    print(torch.add(s, t))
    print(torch.mul(s, s))
