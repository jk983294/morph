#include <ATen/ATen.h>
#include <torch/torch.h>
#include <iostream>

void aten_demo() {
    at::Tensor a = at::ones({2, 2}, at::kInt);
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(at::kInt);
    std::cout << c << std::endl;
}

int main() {
    aten_demo();

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}