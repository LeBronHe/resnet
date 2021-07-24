#ifndef MNIST_HXX
#define MNIST_HXX

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "tensor.hxx"

namespace fs = std::filesystem;

class Mnist {
public:
    Mnist(const fs::path& root);
    Mnist(const Mnist&) = delete;

    ~Mnist();

    [[nodiscard]]
    auto load_train_data() -> std::pair<std::vector<float*>*, std::vector<float*>*>;
    
    [[nodiscard]]
    auto load_test_data() -> std::pair<std::vector<float*>*, std::vector<float*>*>;

    auto shuffle() -> void;

    /*
    template <typename T>
    auto get_tensor(
        const std::vector<float*>* vec,
        std::size_t inner_buf_size,
        int batch_size,
        std::size_t idx
    ) const -> Tensor<T>*;
    */

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
    
    std::vector<float*> train_images;
    std::vector<float*> train_labels;
    std::vector<float*> test_images;
    std::vector<float*> test_labels;

    bool has_train_data = false;
    bool has_test_data = false;

    auto read_header(std::unique_ptr<char[]>& buf, std::size_t idx) const -> std::uint32_t;

    auto read_file(std::vector<float*>& vec, const fs::path& filename) -> void;
};

/*
template <typename T>
auto Mnist::get_tensor(
    const std::vector<float*>* vec,
    std::size_t inner_buf_size,
    int batch_size,
    std::size_t idx
) const -> Tensor<T>* {
    auto tensor = new Tensor<T>(batch_size, 1, 1, 1);

    return tensor;
}
*/

#endif /* MNIST_HXX */
