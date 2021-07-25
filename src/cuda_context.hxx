#ifndef CUDA_CONTEXT_HXX
#define CUDA_CONTEXT_HXX

#include <cublas_v2.h>
#include <cudnn.h>

struct CudaContext {
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;

    CudaContext() {
        cublasCreate_v2(&cublas_handle);
        cudnnCreate(&cudnn_handle);
    }

    CudaContext(const CudaContext&) = delete;

    ~CudaContext() {
        cublasDestroy_v2(cublas_handle);
        cudnnDestroy(cudnn_handle);
    }

    auto operator=(const CudaContext&) = delete;
};

#endif /* CUDA_CONTEXT_HXX */
