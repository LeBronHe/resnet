#include "mnist.hxx"

#include <algorithm>
#include <fstream>
#include <stdexcept>

Mnist::Mnist(const fs::path& root) : root{root} {}

auto Mnist::load_train_data() -> std::pair<std::vector<float>*, std::vector<float>*> {
    read_file(train_images, TRAIN_IMAGES_FILENAME);
    read_file(train_labels, TRAIN_LABELS_FILENAME);

    return std::make_pair(&train_images, &train_labels);
}

auto Mnist::load_test_data() -> std::pair<std::vector<float>*, std::vector<float>*> {
    read_file(test_images, TEST_IMAGES_FILENAME);
    read_file(test_labels, TEST_LABELS_FILENAME);

    return std::make_pair(&test_images, &test_labels);
}

auto Mnist::read_header(std::unique_ptr<char[]>& buf, std::size_t idx) const -> std::uint32_t {
    auto header = reinterpret_cast<std::uint32_t*>(buf.get());
    auto x = *(header + idx);

    return (x >> 24) | ((x & 0x00FF0000) >> 8) | ((x & 0x0000FF00) << 8) | (x << 24);
}

auto Mnist::read_file(std::vector<float>& vec, const fs::path& filename) -> void {
    fs::path path = root / filename;
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open MNIST file");
    }

    auto size = fs::file_size(path);
    auto buf = std::make_unique<char[]>(size);

    file.seekg(0, std::ios::beg);
    file.read(buf.get(), size);
    file.close();

    auto magic_number = read_header(buf, 0);
    auto count = read_header(buf, 1);

    if (magic_number == IMAGE_MAGIC_NUMBER) {
        auto height = read_header(buf, 2);
        auto width = read_header(buf, 3);
        std::size_t pixels = count * height * width;

        vec.reserve(pixels);

        auto ptr = reinterpret_cast<std::uint8_t*>(buf.get() + 16);
        for (std::size_t idx = 0; idx < pixels; ++idx) {
            auto pixel = *ptr++;
            vec.push_back(pixel / 255.0f);
        }
    } else if (magic_number == LABEL_MAGIC_NUMBER) {
        vec.assign(count * CLASSES, 0.0f);

        auto ptr = reinterpret_cast<std::uint8_t*>(buf.get() + 8);
        for (std::size_t idx = 0; idx < count; ++idx) {
            auto label = *ptr++;
            vec[(idx * CLASSES) + label] = 1.0f;
        }
    } else {
        throw std::runtime_error("Invalid MNIST file");
    }
}

auto Mnist::shuffle() -> void {
}
