#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
    torch::Tensor b = torch::randn({2, 2});
    auto c = (a + b).sum();
    c.backward();  // a.grad() will now hold the gradient of c w.r.t. a.
    std::cout << a.grad() << std::endl;
}