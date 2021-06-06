#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

enum class LabelFormat {
    SCALAR,
    VECTOR,
};

struct MnistData {
    std::vector<std::uint8_t> labels;
    std::vector<std::uint8_t> images;
};

class Mnist {
public:
    Mnist() = default;
    Mnist(const Mnist&) = delete;

    ~Mnist() = default;

    constexpr auto set_label_format(LabelFormat label_format) -> void {
        this->label_format = label_format;
    }

    [[nodiscard]]
    auto get_train_data() -> MnistData;
    
    [[nodiscard]]
    auto get_test_data() -> MnistData; 
    
    [[nodiscard]]
    auto normalize_images(const std::vector<uint8_t>& vec) const -> std::vector<float>;

    auto operator=(const Mnist&) -> Mnist& = delete;

private:
    static constexpr std::uint32_t LABEL_MAGIC_NUMBER = 0x0000'0801;
    static constexpr std::uint32_t IMAGE_MAGIC_NUMBER = 0x0000'0803;
    static constexpr std::size_t CLASSES = 10;
    
    static inline const fs::path BASE_PATH{"./data/mnist"};
    static inline const fs::path TRAIN_LABELS_FILENAME{"train-labels-idx1-ubyte"};
    static inline const fs::path TRAIN_IMAGES_FILENAME{"train-images-idx3-ubyte"};
    static inline const fs::path TEST_LABELS_FILENAME{"t10k-labels-idx1-ubyte"};
    static inline const fs::path TEST_IMAGES_FILENAME{"t10k-images-idx3-ubyte"};

    LabelFormat label_format = LabelFormat::SCALAR;

    constexpr auto swap_bytes(std::uint32_t x) const -> std::uint32_t {
        x = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0xFF00FF);
        return ((x << 16) | (x >> 16));
    }

    [[nodiscard]]
    auto load_labels(const fs::path& path) -> std::vector<std::uint8_t>;

    [[nodiscard]]
    auto load_images(const fs::path& path) -> std::vector<std::uint8_t>;
};
