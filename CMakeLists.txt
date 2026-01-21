cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME yolov8)
project(${PROJECT_NAME})

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/daway/local/TensorRT-8.2.4.2/include)
link_directories(/home/daway/local/TensorRT-8.2.4.2/lib)

# opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS 
    ${PROJECT_SOURCE_DIR}/src/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/*.cu 
    ${PROJECT_SOURCE_DIR}/demo.cpp)
add_executable(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nvinfer cudart)
