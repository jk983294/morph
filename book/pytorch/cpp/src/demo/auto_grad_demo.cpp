#include <torch/csrc/autograd/variable.h>
#include <torch/torch.h>
#include <iostream>

void compute_higher_order_gradients_example();

int main() {
    std::cout << std::boolalpha;
    torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
    std::cout << a << std::endl;

    auto y = a + 2;
    std::cout << y << std::endl;
    std::cout << y.grad_fn()->name() << std::endl;  // AddBackward1

    auto z = y * y * 3;
    auto out = z.mean();
    std::cout << z << std::endl;
    std::cout << z.grad_fn()->name() << std::endl;  // MulBackward1
    std::cout << out << std::endl;
    std::cout << out.grad_fn()->name() << std::endl;  // MeanBackward0
    out.backward();
    std::cout << a.grad() << std::endl;

    // Now let's take a look at an example of vector-Jacobian product:
    auto x = torch::randn(3, torch::requires_grad());

    y = x * 2;
    while (y.norm().item<double>() < 1000) {
        y = y * 2;
    }

    std::cout << y << std::endl;
    std::cout << y.grad_fn()->name() << std::endl;  // MulBackward1

    auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
    y.backward(v);
    std::cout << x.grad() << std::endl;

    // stop autograd from tracking history on tensors
    std::cout << x.requires_grad() << std::endl;
    std::cout << x.pow(2).requires_grad() << std::endl;

    {
        torch::NoGradGuard no_grad;
        std::cout << x.pow(2).requires_grad() << std::endl;  // false
    }
    std::cout << x.requires_grad() << std::endl;  // true

    compute_higher_order_gradients_example();
}

void compute_higher_order_gradients_example() {
    std::cout << "====== Running \"Computing higher-order gradients in C++\" ======" << std::endl;

    // one of the applications of higher-order gradients is calculating gradient penalty
    auto model = torch::nn::Linear(4, 3);

    auto input = torch::randn({3, 4}).requires_grad_(true);
    auto output = model(input);

    // calculate loss
    auto target = torch::randn({3, 3});
    auto loss = torch::nn::MSELoss()(output, target);

    // use norm of gradients as penalty
    auto grad_output = torch::ones_like(output);
    auto gradient = torch::autograd::grad({output}, {input}, {grad_output}, true)[0];
    auto gradient_penalty = torch::pow((gradient.norm(2, 1) - 1), 2).mean();

    // add gradient penalty to loss
    auto combined_loss = loss + gradient_penalty;
    combined_loss.backward();

    std::cout << input.grad() << std::endl;
}