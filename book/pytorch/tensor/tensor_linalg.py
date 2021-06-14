import torch


def inverse_demo():
    x = torch.rand(4, 4)
    y = torch.inverse(x)
    z = torch.mm(x, y)
    print(z)


def eigen_demo():
    a = torch.diag(torch.tensor([1, 2, 3], dtype=torch.double))
    e, v = torch.eig(a, eigenvectors=True)
    print(e)
    print(v)


def det_demo():
    a = torch.randn(3, 3)
    print(torch.linalg.det(a))


def cholesky_demo():
    a = torch.randn(3, 3)
    a = torch.mm(a, a.t())  # make symmetric positive-definite
    l = torch.cholesky(a)
    print(a)
    print(l)
    print(torch.mm(l, l.t()))  # a = l * l^T

    a = torch.randn(3, 3)
    a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3)  # make symmetric positive definite
    u = torch.cholesky(a)
    print(a)
    print(u)
    print(torch.cholesky_inverse(u))
    print(a.inverse())

    b = torch.randn(3, 2)
    print(b)
    print(torch.cholesky_solve(b, u))
    print(torch.mm(a.inverse(), b))


def least_square_demo():
    A = torch.tensor([[1., 1, 1], [2, 3, 4], [3, 5, 2], [4, 2, 5], [5, 4, 3]])
    B = torch.tensor([[-10., -3], [12, 14], [14, 12], [16, 16], [18, 16]])
    X, _ = torch.lstsq(B, A)
    print(X)


def solve_demo():
    A = torch.tensor([[6.80, -2.11, 5.66, 5.97, 8.23],
                      [-6.05, -3.30, 5.36, -4.44, 1.08],
                      [-0.45, 2.58, -2.70, 0.27, 9.04],
                      [8.32, 2.71, 4.35, -7.17, 2.14],
                      [-9.67, -5.14, -7.26, 6.08, -6.87]]).t()
    B = torch.tensor([[4.02, 6.19, -8.22, -7.57, -3.03],
                      [-1.56, 4.00, -8.67, 1.75, 2.86],
                      [9.81, -4.09, -4.57, -8.61, 8.99]]).t()
    X, LU = torch.solve(B, A)
    print(X)  # AX = B


if __name__ == '__main__':
    inverse_demo()
    eigen_demo()
    det_demo()
    cholesky_demo()
    least_square_demo()
    solve_demo()
