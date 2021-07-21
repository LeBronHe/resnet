#ifndef TENSOR_HXX
#define TENSOR_HXX

#include <cudnn.h>

enum class DeviceType {
    Host,
    Cuda
};

template <typename T>
class Tensor {
public:
    T *ptr;
    T *d_ptr;
    int n;
    int c;
    int h;
    int w;

public:
    Tensor(int n, int c, int h, int w, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW);
    Tensor(const Tensor&) = delete;

    ~Tensor();

    constexpr auto len() const { return n * c * h * w; }

    constexpr auto size() const { return len() * sizeof(T); }

    auto get_desc() -> cudnnTensorDescriptor_t;

    auto memcpy(DeviceType device) -> T*;

    auto print(int precision = 3) -> void;

    auto operator=(const Tensor&) = delete;

private:
    cudnnTensorDescriptor_t desc = nullptr;
    cudnnTensorFormat_t format;
};

#endif /* TENSOR_HXX */
