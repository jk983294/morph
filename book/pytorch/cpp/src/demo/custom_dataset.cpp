#include <torch/torch.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

const int image_size = 224;
const size_t train_batch_size = 8;
const size_t test_batch_size = 200;
const size_t iterations = 10;
const size_t log_interval = 20;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
    using Example = torch::data::Example<>;

    vector<long> data;

public:
    explicit CustomDataset(const vector<long>& data) : data(data) {}

    Example get(size_t index) override {
        auto tdata = torch::rand({3, image_size, image_size});
        auto tlabel = torch::from_blob(&data[index], {1}, torch::kLong);
        return {tdata, tlabel};
    }

    torch::optional<size_t> size() const override { return data.size(); }
};

std::pair<vector<long>, vector<long>> readInfo() {
    vector<long> train, test;

    random_device rd;
    mt19937 generator(rd());
    std::uniform_int_distribution<long> uid(0, 9);

    for (int i = 0; i < 1000; ++i) {
        train.emplace_back(uid(generator));
    }
    for (int i = 0; i < 100; ++i) {
        test.emplace_back(uid(generator));
    }

    std::random_shuffle(train.begin(), train.end());
    std::random_shuffle(test.begin(), test.end());
    return std::make_pair(train, test);
}

struct NetworkImpl : torch::nn::SequentialImpl {
    NetworkImpl() {
        using namespace torch::nn;

        auto stride = torch::ExpandingArray<2>({2, 2});
        torch::ExpandingArray<2> shape({-1, 256 * 6 * 6});
        push_back(Conv2d(Conv2dOptions(3, 64, 11).stride(4).padding(2)));
        push_back(Functional(torch::relu));
        push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
        push_back(Conv2d(Conv2dOptions(64, 192, 5).padding(2)));
        push_back(Functional(torch::relu));
        push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
        push_back(Conv2d(Conv2dOptions(192, 384, 3).padding(1)));
        push_back(Functional(torch::relu));
        push_back(Conv2d(Conv2dOptions(384, 256, 3).padding(1)));
        push_back(Functional(torch::relu));
        push_back(Conv2d(Conv2dOptions(256, 256, 3).padding(1)));
        push_back(Functional(torch::relu));
        push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
        push_back(Functional(torch::reshape, shape));
        push_back(Dropout());
        push_back(Linear(256 * 6 * 6, 4096));
        push_back(Functional(torch::relu));
        push_back(Dropout());
        push_back(Linear(4096, 4096));
        push_back(Functional(torch::relu));
        push_back(Linear(4096, 102));
        push_back(Functional(torch::nn::functional::log_softmax, torch::nn::functional::LogSoftmaxFuncOptions(1)));
    }
};
TORCH_MODULE(Network);

template <typename DataLoader>
void train(Network& network, DataLoader& loader, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size,
           torch::DeviceType device) {
    size_t index = 0;
    network->train();
    float Loss = 0, Acc = 0;

    for (auto& batch : loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device).view({-1});

        auto output = network->forward(data);
        auto loss = torch::nll_loss(output, targets);
        assert(!std::isnan(loss.template item<float>()));
        auto acc = output.argmax(1).eq(targets).sum();

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();

        if (index++ % log_interval == 0) {
            auto end = std::min(data_size, (index + 1) * train_batch_size);

            std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size << "\tLoss: " << Loss / end
                      << "\tAcc: " << Acc / end << std::endl;
        }
    }
}

template <typename DataLoader>
void test(Network& network, DataLoader& loader, size_t data_size, torch::DeviceType device) {
    size_t index = 0;
    network->eval();
    torch::NoGradGuard no_grad;
    float Loss = 0, Acc = 0;

    for (const auto& batch : loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device).view({-1});

        auto output = network->forward(data);
        auto loss = torch::nll_loss(output, targets);
        assert(!std::isnan(loss.template item<float>()));
        auto acc = output.argmax(1).eq(targets).sum();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }

    if (index++ % log_interval == 0)
        std::cout << "Test Loss: " << Loss / data_size << "\tAcc: " << Acc / data_size << std::endl;
}

int main() {
    torch::manual_seed(1);

    torch::DeviceType device = torch::kCPU;
    if (torch::cuda::is_available()) device = torch::kCUDA;
    std::cout << "Running on: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    auto data = readInfo();

    auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
    auto train_size = train_set.size().value();
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_set), train_batch_size);

    auto test_set = CustomDataset(data.second).map(torch::data::transforms::Stack<>());
    auto test_size = test_set.size().value();
    auto test_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_set), test_batch_size);

    Network network;
    network->to(device);

    torch::optim::SGD optimizer(network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

    for (size_t i = 0; i < iterations; ++i) {
        train(network, *train_loader, optimizer, i + 1, train_size, device);
        std::cout << std::endl;
        test(network, *test_loader, test_size, device);
        std::cout << std::endl;
    }

    return 0;
}