import torch

if __name__ == '__main__':
    a = torch.empty(3, 3).uniform_(0, 1.)
    print(a)
    print(torch.bernoulli(a))  # draw binary random numbers (0 or 1) from a Bernoulli distribution
    weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)  # create a tensor of weights
    print(torch.multinomial(weights, 2))

    x = torch.randn(3)
    print(x)
    print(torch.mean(x))
    print(torch.sum(x))
    print(torch.median(x))
    # print(torch.nanmedian(x))  # ignoring NaN values
    print(torch.min(x))
    print(torch.max(x))
    print(torch.mode(x))
    print(torch.std(x))
    print(torch.var(x))
    print(torch.quantile(x, 0.1))
    print(x.nanquantile(0.1))
    print(torch.nansum(x))  # treating NaN as zero
    print(torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long)))

    # count the frequency of each value in an array of non-negative int
    print(torch.bincount(torch.randint(0, 8, (5,), dtype=torch.int64)))
    print(torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3))

    x = torch.zeros(3, 3)
    x[torch.randn(3, 3) > 0.5] = 1
    print(torch.count_nonzero(x))
