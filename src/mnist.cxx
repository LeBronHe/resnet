#include "mnist.hxx"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <iostream>

namespace resnet {

auto Mnist::load_labels(const fs::path& path) -> std::vector<uint8_t> {
    std::vector<uint8_t> buffer;
    
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);

    if (file.is_open()) {
        std::uint32_t magic_number;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = swap_bytes(magic_number);

        if (magic_number != LABEL_MAGIC_NUMBER) {
            std::string message = "Invalid MNIST label file: " + path.string();
            throw std::runtime_error(message);
        }

        std::uint32_t items_count;
        file.read(reinterpret_cast<char*>(&items_count), sizeof(items_count));
        items_count = swap_bytes(items_count);

        if (label_format == LabelFormat::SCALAR) {
            buffer.reserve(items_count);
        } else {
            buffer.assign(items_count * CLASSES, 0);
        }

        for (std::size_t idx = 0; idx < items_count; ++idx) {
            std::uint8_t label;
            file.read(reinterpret_cast<char*>(&label), sizeof(label));

            if (label_format == LabelFormat::SCALAR) {
                buffer.push_back(label);
            } else {
                buffer[(idx * CLASSES) + label] = 1;
            }
        }

        file.close();
    }

    return buffer;
}

}
