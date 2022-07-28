#include <iostream>
#include <torch/torch.h>

struct Net1 : torch::nn::Module{
    Net1(int64_t N, int64_t M){
        W = register_parameter("W", torch::randn({N, M}));
        b = register_parameter("b", torch::rand(M));
    }
    torch::Tensor forward(torch::Tensor input){
        return torch::addmm(b, input, W);
    }
    torch::Tensor W, b;
};

struct Net2 : torch::nn::Module {
    Net2(int64_t N, int64_t M)
        : linear(register_module("linear", torch::nn::Linear(N, M))) {
            another_bias = register_parameter("b", torch::randn(M));
    }
    torch::Tensor forward(torch::Tensor input){
        return linear(input) + another_bias;
    }
    torch::nn::Linear linear;
    torch::Tensor another_bias;
};

int main(){
    // torch::Tensor tensor = torch::eye(3);
    // std::cout << tensor << std::endl;
    Net2 net(4, 5);
    // for (const auto& p : net.parameters()){
    //     std::cout << p << std::endl;
    // }
    // for (const auto& pair : net.named_parameters()){
    //     std::cout << pair.key() << ": " << pair.value() << std::endl;
    // }

    std::cout << net.forward(torch::ones({2,4})) << std::endl;

    return 0;
}