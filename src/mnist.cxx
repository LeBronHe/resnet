#include "mnist.hxx"

#include <fstream>
#include <stdexcept>

#include <iostream>

namespace resnet {

Mnist::Mnist() {
}

Mnist::~Mnist() {
}

auto Mnist::load_labels(const fs::path& filename) -> void {
    std::ifstream file(BASE_PATH / filename, std::ios::in | std::ios::binary);

    if (file.is_open()) {
        std::uint32_t magic_number;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = swap_bytes(magic_number);

        if (magic_number != LABEL_MAGIC_NUMBER) {
            throw std::runtime_error("Invalid MNIST file");
        }

        std::uint32_t number_of_items;
        file.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));
        number_of_items = swap_bytes(number_of_items);

        training_labels.reserve(number_of_items);

        for (std::size_t i = 0; i < number_of_items; i++) {
            std::uint8_t label;
            file.read(reinterpret_cast<char*>(&label), sizeof(label));
            training_labels.push_back(label);
        }
    }

    for (const auto& label : training_labels) {
        std::cout << +label;
    }

    std::cout << "\n";

    file.close();
}

}
