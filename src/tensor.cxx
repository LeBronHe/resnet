#include "tensor.hxx"

#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>

template <typename T>
Tensor<T>::Tensor(int n, int c, int h, int w, cudnnTensorFormat_t format)
    : n(n), c(c), h(h), w(w), format(format) {
    ptr = new T[len()]();
    cudaMalloc(&d_ptr, size());
}

template <typename T>
Tensor<T>::~Tensor() {
    delete[] ptr;
    cudaFree(d_ptr);
    cudnnDestroyTensorDescriptor(desc);
}

template <typename T>
auto Tensor<T>::get_desc() -> cudnnTensorDescriptor_t {
    if (desc != nullptr) {
        return desc;
    }

    cudnnDataType_t data_type;
    switch (sizeof(T)) {
        case 2:
            data_type = CUDNN_DATA_HALF;
            break;
        case 4:
            data_type = CUDNN_DATA_FLOAT;
            break;
        default:
            throw std::runtime_error("Unsupported data type");
    }
    
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, format, data_type, n, c, h, w);

    return desc;
};

template <typename T>
auto Tensor<T>::memcpy(DeviceType device) -> T* {
    if (device == DeviceType::Host) {
        cudaMemcpy(ptr, d_ptr, size(), cudaMemcpyDeviceToHost);
        return ptr;
    } else {
        cudaMemcpy(d_ptr, ptr, size(), cudaMemcpyHostToDevice);
        return d_ptr;
    }
}

template <typename T>
auto Tensor<T>::print(int precision) -> void {
    std::cout << "Tensor[" << n << ", " << c << ", " << h << ", " << w << "]\n";
    std::cout << "Memory size: " << size() << " bytes\n";
    std::cout << std::hex << "Host: " << ptr << ", Device: " << d_ptr << std::dec << std::endl;

    for (std::size_t idx = 0; idx < len(); ++idx) {
        if (idx % 16 == 0 && idx != 0) {
            std::cout << "\n";
        }

        std::cout << std::fixed << std::setprecision(precision) << ptr[idx] << " ";
    }

    std::cout << "\n";
    std::cout.unsetf(std::ios::fixed);
}

template class Tensor<half>;
template class Tensor<float>;
