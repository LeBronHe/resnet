cmake_minimum_required(VERSION 3.20)

project(resnet CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# Latest supported standard by CUDA Toolkit 11.2
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(CHECK_START "Downloading MNIST database")
execute_process(COMMAND python3 download_mnist.py ERROR_VARIABLE DOWNLOAD_ERROR)
if (DOWNLOAD_ERROR)
  message(FATAL_ERROR "Failed to download MNIST database: ${DOWNLOAD_ERROR}")
endif()
message(CHECK_PASS "done")

add_executable(${PROJECT_NAME} src/main.cxx)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cublas cudnn)

add_subdirectory(src)
