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
    
    auto [ train_labels, train_images ] = mnist.get_train_data();
    
    for (std::size_t idx = 0; idx < 100; idx++) {
        if (idx % 10 == 0 && idx != 0) {
            std::cout << std::endl;
        }

        std::cout << +train_labels[idx] << " ";
    }

    std::cout << std::endl;
    
    return 0;
}
