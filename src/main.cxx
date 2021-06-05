#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

static uint32_t LABEL_MAGIC_NUMBER = 0x0000'0801;
static uint32_t IMAGE_MAGIC_NUMBER = 0x0000'0803;

auto main(int argc, char** argv) -> int {
    std::vector<std::string_view> args(argv + 1, argv + argc);

    for (const auto& arg : args) {
        std::cout << arg << std::endl;
    }

    std::ifstream file("./data/mnist/train-labels-idx1-ubyte", std::ios::in | std::ios::binary);

    if (file.is_open()) {
        auto swap_bytes = [](uint32_t value) {
            value = ((value << 8) & 0xFF00FF00) | ((value >> 8) & 0xFF00FF);
            return (value << 16) | (value >> 16);
        };

        uint32_t magic_number = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = swap_bytes(magic_number);

        uint32_t number_of_items;
        file.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));
        number_of_items = swap_bytes(number_of_items);

        std::cout << magic_number << std::endl;
        std::cout << number_of_items << std::endl;
        
        for (size_t i = 0; i < 10; i++) {
            char label = 0;
            file.read(&label, sizeof(label));

            std::cout << std::to_string(label) << std::endl;
        }

        file.close();
    }

    return 0;
}
