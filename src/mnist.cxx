#include "mnist.hxx"

#include <fstream>
#include <stdexcept>

#include <iostream>

auto Mnist::get_train_data() -> MnistData {
    MnistData train_data {
        load_labels(BASE_PATH / TRAIN_LABELS_FILENAME),
        load_images(BASE_PATH / TRAIN_IMAGES_FILENAME)
    };

    return train_data;
}

auto Mnist::get_test_data() -> MnistData {
    MnistData test_data {
        load_labels(BASE_PATH / TEST_LABELS_FILENAME),
        load_images(BASE_PATH / TEST_IMAGES_FILENAME)
    };

    return test_data;
}

auto Mnist::normalize_images(const std::vector<uint8_t>& vec) const -> std::vector<float> {
    std::vector<float> _vec(vec.size());

    std::transform(vec.begin(), vec.end(), _vec.begin(), [](const auto& pixel) {
        return pixel / 255.0f;
    });

    return _vec;
}

auto Mnist::load_labels(const fs::path& path) -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> vec;
    
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open MNIST labels file");
    }

    std::uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap_bytes(magic_number);

    if (magic_number != LABEL_MAGIC_NUMBER) {
        throw std::runtime_error("Invalid MNIST labels file");
    }

    std::uint32_t items_count;
    file.read(reinterpret_cast<char*>(&items_count), sizeof(items_count));
    items_count = swap_bytes(items_count);

    if (label_format == LabelFormat::SCALAR) {
        vec.reserve(items_count);
    } else {
        vec.assign(items_count * CLASSES, 0);
    }

    for (std::size_t idx = 0; idx < items_count; idx++) {
        std::uint8_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));

        if (label_format == LabelFormat::SCALAR) {
            vec.push_back(label);
        } else {
            vec[(idx * CLASSES) + label] = 1;
        }
    }

    file.close();

    return vec;
}

auto Mnist::load_images(const fs::path& path) -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> vec;

    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open MNIST images file");
    }

    std::uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap_bytes(magic_number);

    if (magic_number != IMAGE_MAGIC_NUMBER) {
        throw std::runtime_error("Invalid MNIST images file");
    }

    std::uint32_t images_count;
    file.read(reinterpret_cast<char*>(&images_count), sizeof(images_count));
    images_count = swap_bytes(images_count);

    // File has two 32-bit integers for image dimensions
    // Not used for anything here so read and forget
    std::uint32_t tmp;
    file.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));
    file.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));

    for (std::size_t idx = 0; idx < images_count; idx++) {
        std::uint8_t pixel;
        file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));

        vec.push_back(pixel);
    }

    file.close();

    return vec;
}
