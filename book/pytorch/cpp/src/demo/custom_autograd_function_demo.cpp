#include <torch/torch.h>
#include <iostream>

using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

class LinearFunction : public torch::autograd::Function<LinearFunction> {
public:
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input, torch::Tensor weight,
                                 torch::Tensor bias = torch::Tensor()) {
        ctx->save_for_backward({input, weight, bias});
        auto output = input.mm(weight.t());
        if (bias.defined()) {
            output += bias.unsqueeze(0).expand_as(output);
        }
        return output;
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];

        auto grad_output = grad_outputs[0];
        auto grad_input = grad_output.mm(weight);
        auto grad_weight = grad_output.t().mm(input);
        auto grad_bias = torch::Tensor();
        if (bias.defined()) {
            grad_bias = grad_output.sum(0);
        }
        return {grad_input, grad_weight, grad_bias};
    }
};

class MulConstant : public torch::autograd::Function<MulConstant> {
public:
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor tensor, double constant) {
        ctx->saved_data["constant"] = constant;
        return tensor * constant;
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
    }
};

int main() {
    std::cout << "====== Running \"Using custom autograd function in C++\" ======" << std::endl;
    {
        auto x = torch::randn({2, 3}).requires_grad_();
        auto weight = torch::randn({4, 3}).requires_grad_();
        auto y = LinearFunction::apply(x, weight);
        y.sum().backward();

        std::cout << x.grad() << std::endl;
        std::cout << weight.grad() << std::endl;
    }

    {
        auto x = torch::randn({2}).requires_grad_();
        auto y = MulConstant::apply(x, 5.5);
        y.sum().backward();

        std::cout << x.grad() << std::endl;
    }
}