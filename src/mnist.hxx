#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace resnet {

class Mnist {
public:
    Mnist();
    Mnist(const Mnist&) = delete;

    ~Mnist();

    auto operator=(const Mnist&) -> Mnist& = delete;

private:
    static const std::uint32_t LABEL_MAGIC_NUMBER = 0x0000'0801;
    static const std::uint32_t IMAGE_MAGIC_NUMBER = 0x0000'0803;
    static inline const fs::path BASE_PATH{"./data/mnist"};
    static inline const fs::path TRAINING_LABELS_FILENAME{"train-labels-idx1-ubyte"};
    
    std::vector<std::uint8_t> training_labels;

    inline auto swap_bytes(std::uint32_t x) -> std::uint32_t {
        x = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0xFF00FF);
        return ((x << 16) | (x >> 16));
    }

    auto load_labels(const fs::path& filename) -> void;
};

}
