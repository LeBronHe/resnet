#include "mnist.hxx"

#include <algorithm>
#include <fstream>
#include <random>
#include <stdexcept>

Mnist::Mnist(const fs::path& root) : root{root} {}

Mnist::~Mnist() {
    if (has_train_data) {
        for (std::size_t idx = 0; idx < 60'000; ++idx) {
            delete[] train_images[idx];
            delete[] train_labels[idx];
        }
    }

    if (has_test_data) {
        for (std::size_t idx = 0; idx < 10'000; ++idx) {
            delete[] test_images[idx];
            delete[] test_labels[idx];
        }
    }
}

auto Mnist::load_train_data() -> std::pair<std::vector<float*>*, std::vector<float*>*> {
    read_file(train_images, TRAIN_IMAGES_FILENAME);
    read_file(train_labels, TRAIN_LABELS_FILENAME);

    has_train_data = true;

    return std::make_pair(&train_images, &train_labels);
}

auto Mnist::load_test_data() -> std::pair<std::vector<float*>*, std::vector<float*>*> {
    read_file(test_images, TEST_IMAGES_FILENAME);
    read_file(test_labels, TEST_LABELS_FILENAME);

    has_test_data = true;

    return std::make_pair(&test_images, &test_labels);
}

auto Mnist::read_header(std::unique_ptr<char[]>& buf, std::size_t idx) const -> std::uint32_t {
    auto header = reinterpret_cast<std::uint32_t*>(buf.get());
    auto x = *(header + idx);

    return (x >> 24) | ((x & 0x00FF0000) >> 8) | ((x & 0x0000FF00) << 8) | (x << 24);
}

auto Mnist::read_file(std::vector<float*>& vec, const fs::path& filename) -> void {
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

    vec.reserve(count);
    
    if (magic_number == IMAGE_MAGIC_NUMBER) {
        height = read_header(buf, 2);
        width = read_header(buf, 3);

        auto ptr = reinterpret_cast<std::uint8_t*>(buf.get() + 16);
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t pixels = height * width;
            float* image = new float[pixels];

            for (std::size_t j = 0; j < pixels; ++j) {
                auto pixel = *ptr++;
                image[j] = pixel / 255.0f;
            }

            vec.emplace_back(image);
        }
    } else if (magic_number == LABEL_MAGIC_NUMBER) {
        auto ptr = reinterpret_cast<std::uint8_t*>(buf.get() + 8);
        for (std::size_t i = 0; i < count; ++i) {
            auto label = *ptr++;
            float* one_hot_vec = new float[CLASSES]();

            one_hot_vec[label] = 1.0f;

            vec.emplace_back(one_hot_vec);
        }
    } else {
        throw std::runtime_error("Invalid MNIST file");
    }
}

auto Mnist::shuffle() -> void {
    std::random_device rd;
    std::mt19937 g_images(rd());
    auto g_labels = g_images;

    std::shuffle(std::begin(train_images), std::end(train_images), g_images);
    std::shuffle(std::begin(train_labels), std::end(train_labels), g_labels);
}
