#include <iostream>
#include <string_view>
#include <vector>

#include "mnist.hxx"

auto main(int argc, char** argv) -> int {
    std::vector<std::string_view> args(argv + 1, argv + argc);

    for (const auto& arg : args) {
        std::cout << arg << std::endl;
    }

    resnet::Mnist mnist(resnet::LabelFormat::VECTOR);

    std::vector<uint8_t> test_labels = mnist.load_labels("./data/mnist/train-labels-idx1-ubyte");

    for (std::size_t i = 0; i < 100; ++i) {
        if (i % 10 == 0 && i != 0) {
            std::cout << std::endl;
        }
        
        std::cout << +test_labels[i] << " ";
    }

    std::cout << test_labels.size() << std::endl;

    return 0;
}
