#include <iostream>
#include <string_view>
#include <vector>

#include "mnist.hxx"

auto main(int argc, char** argv) -> int {
    std::vector<std::string_view> args(argv + 1, argv + argc);

    for (const auto& arg : args) {
        std::cout << arg << std::endl;
    }

    Mnist mnist;
    mnist.set_label_format(LabelFormat::VECTOR);

    //std::vector<uint8_t> test_labels = mnist.load_labels("./data/mnist/train-labels-idx1-ubyte");
    
    std::vector<uint8_t> train_images = mnist.load_images("./data/mnist/train-images-idx3-ubyte");

    for (std::size_t idx = 0; idx < 28 * 28 * 10; idx++) {
        if (idx % 28 == 0 && idx != 0) {
            std::cout << std::endl;
        }

        std::cout << +train_images[idx] << " ";
    }

    std::cout << train_images.size() << std::endl;

    return 0;
}
