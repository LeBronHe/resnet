#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace resnet {

static std::size_t CLASSES = 10;

enum class LabelFormat {
    SCALAR,
    VECTOR,
};

class Mnist {
public:
    Mnist() = default;
    Mnist(LabelFormat label_format) : label_format(label_format) {}
    Mnist(const Mnist&) = delete;

    ~Mnist() = default;

    auto operator=(const Mnist&) -> Mnist& = delete;

    static constexpr std::uint32_t LABEL_MAGIC_NUMBER = 0x0000'0801;
    static constexpr std::uint32_t IMAGE_MAGIC_NUMBER = 0x0000'0803;

    static inline const fs::path BASE_PATH{"./data/mnist"};
    static inline const fs::path TRAINING_LABELS_FILENAME{"train-labels-idx1-ubyte"};
    static inline const fs::path TRAINING_IMAGES_FILENAME{"train-images-idx3-ubyte"};
    static inline const fs::path TEST_LABELS_FILENAME{"t10k-labels-idx1-ubyte"};
    static inline const fs::path TEST_IMAGES_FILENAME{"t10k-images-idx3-ubyte"};

    LabelFormat label_format = LabelFormat::SCALAR;
    
    std::vector<std::uint8_t> training_labels;
    std::vector<std::uint8_t> training_images;

    std::vector<std::uint8_t> test_labels;
    std::vector<std::uint8_t> test_images;

    inline auto swap_bytes(std::uint32_t x) -> std::uint32_t {
        x = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0xFF00FF);
        return ((x << 16) | (x >> 16));
    }

    [[nodiscard]]
    auto load_labels(const fs::path& path) -> std::vector<uint8_t>;
};

}
