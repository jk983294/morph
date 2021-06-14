import torch
import numpy as np

if __name__ == '__main__':
    x = torch.randn(3, dtype=torch.float)
    M = torch.randn(3, 5)
    mat1 = torch.randn(3, 3)
    mat2 = torch.randn(3, 5)
    torch.addmm(M, mat1, mat2)
    t = torch.randn(1, 3)
    t1 = torch.randn(3, 1)
    t2 = torch.randn(1, 3)
    print(x)
    print(x.abs())
    print(x.neg())
    print(x.acos())
    print(torch.ceil(x))
    print(torch.floor(x))
    print(torch.round(x))
    # print(torch.diff(x))
    print(torch.clamp(x, min=-0.5, max=0.5))
    print(torch.clamp(x, min=0.5))
    print(torch.clamp(x, max=0.5))
    print(torch.trunc(x))  # truncated integer values
    print(torch.frac(x))  # fractional portion of each element
    print(x.add(1))
    print(torch.exp(x))
    print(torch.expm1(x))
    print(torch.logit(x))
    print(torch.mul(x, 100))
    print(torch.addcdiv(t, t1, t2, value=0.1))  # t + value * t1 / t2
    print(torch.addcmul(t, t1, t2, value=0.1))  # t + value * t1 * t2
    print(torch.addmm(M, mat1, mat2))  # beta * M + alpha * mat1 * mat2
    print(torch.matmul(mat1, mat2))  # mat1 * mat2
    print(torch.mm(mat1, mat2))  # mat1 * mat2
    print(torch.matrix_power(mat1, 2))  # mat1 * mat1
    print(torch.addmv(x, mat1, x))  # β x+α (mat * x)
    print(torch.mv(mat1, x))  # mat * vec
    print(torch.outer(x, x))  # vec1⊗vec2
    print(torch.renorm(mat1, 1, 0, 5))
    input_ = torch.tensor([10000., 1e-07])
    other_ = torch.tensor([10000.1, 1e-08])
    print(torch.floor_divide(input_, other_))  # trunc(input_ / other_)
    print(torch.allclose(input_, other_))  # ∣input−other∣≤atol+rtol×∣other∣
    print(torch.isclose(input_, other_))  # ∣input−other∣≤atol+rtol×∣other∣
    print(mat1)
    print(torch.where(mat1 > 0, mat1, -mat1))
    print(torch.amax(mat1, 0))  # 按列
    print(torch.amax(mat1, 1))  # 按行
    print(torch.max(mat1, 0))  # 按列
    print(torch.max(mat1, 1))  # 按行
    print(torch.argmax(mat1))  # 所有元素
    print(torch.argmax(mat1, 0))  # 按列
    print(torch.argmax(mat1, 1))  # 按行
    print(torch.amin(mat1, 0))  # 按列
    print(torch.amin(mat1, 1))  # 按行
    print(torch.argmin(mat1))  # 所有元素
    print(torch.argmin(mat1, 0))  # 按列
    print(torch.argmin(mat1, 1))  # 按行
    print(torch.argsort(mat1, 0))  # 按列, returns the indices
    print(torch.argsort(mat1, 1))  # 按行
    print(torch.topk(mat1, 2))
    # print(torch.msort(mat1))  # 按行
    print(torch.kthvalue(mat1, 1, 0))
    print(torch.kthvalue(mat1, 1, 1))
    print(torch.logsumexp(mat1, 1))  # 按行

    """cum"""
    print("cum function:")
    print(torch.logcumsumexp(x, dim=0))  # log (sigma(exp(xi)))
    print(torch.cummax(x, dim=0))
    print(torch.cummin(x, dim=0))
    print(torch.cumprod(x, dim=0))
    print(torch.cumsum(x, dim=0))

    """vec <> vec"""
    a = torch.tensor([9.7, float('nan'), 3.1, float('nan')])
    b = torch.tensor([-2.2, 0.5, float('nan'), float('nan')])
    c = torch.tensor([9.7, 1, 3.1, 4])
    d = torch.tensor([1.7, 1.2, 3.1, 2])
    print(torch.maximum(a, b))
    print(torch.minimum(a, b))
    print(torch.fmod(a, 2))
    print(torch.dist(c, d, 1))  # p-norm
    print(torch.norm(c))
    print(torch.div(c, d))
    print(torch.true_divide(c, d))  # rounding_mode=None
    print(torch.sub(c, d, alpha=2.))
    print(c.add(d))
    print(torch.dot(c, d))
    print(torch.sigmoid(c))
    # print(torch.inner(c, d))

    """flip"""
    x = torch.arange(4).view(2, 2)
    print(torch.flipud(x))
    print(torch.fliplr(x))

    # logical
    print("logical function:")
    print(torch.eq(c, d))
    print(torch.ne(c, d))
    print(torch.gt(c, d))
    print(torch.logical_and(c, d))
    print(torch.logical_or(c, d))
    print(torch.logical_xor(c, d))
    print(torch.logical_not(c))
    print(torch.equal(c, d))  # if all equal
    a = torch.rand(2, 2).bool()
    print(a)
    print(torch.all(a))
    print(torch.all(a, dim=0))  # 按列
    print(torch.all(a, dim=1))  # 按行
    print(torch.any(a))
    print(torch.any(a, dim=0))  # 按列
    print(torch.any(a, dim=1))  # 按行
    to_test = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
    print(torch.isfinite(to_test))
    print(torch.isinf(to_test))
    print(torch.isposinf(to_test))
    print(torch.isneginf(to_test))
    print(torch.isnan(to_test))
