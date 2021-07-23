#ifndef MNIST_HXX
#define MNIST_HXX

#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

class Mnist {
public:
    using Vector2d = std::vector<std::unique_ptr<float[]>>;

    Mnist(const fs::path& root);
    Mnist(const Mnist&) = delete;

    ~Mnist() = default;

    [[nodiscard]]
    auto load_train_data() -> std::pair<Vector2d*, Vector2d*>;
    
    [[nodiscard]]
    auto load_test_data() -> std::pair<Vector2d*, Vector2d*>;

    auto shuffle() -> void;

    auto operator=(const Mnist&) = delete;

private:
    static constexpr std::uint32_t IMAGE_MAGIC_NUMBER = 0x0000'0803;
    static constexpr std::uint32_t LABEL_MAGIC_NUMBER = 0x0000'0801;
    static constexpr std::size_t CLASSES = 10;

    static constexpr const char* TRAIN_IMAGES_FILENAME = "train-images-idx3-ubyte";
    static constexpr const char* TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte";
    static constexpr const char* TEST_IMAGES_FILENAME = "t10k-images-idx3-ubyte";
    static constexpr const char* TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte";
    fs::path root;
    
public:
    Vector2d train_images;
    Vector2d train_labels;
    Vector2d test_images;
    Vector2d test_labels;

    auto read_header(std::unique_ptr<char[]>& buf, std::size_t idx) const -> std::uint32_t;

    auto read_file(Vector2d& vec, const fs::path& filename) -> void;
};

#endif /* MNIST_HXX */
