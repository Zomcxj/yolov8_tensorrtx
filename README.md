# YOLOv8 TensorRT 集成

## 项目概述

本项目是基于 [tensorrtx](https://github.com/wang-xinyu/tensorrtx) 项目的 YOLOv8 实现，集成了多种任务类型到一个统一的类中，包括目标检测 (det)、语义分割 (seg)、人体姿态估计 (pose)、旋转框目标检测 (obb) 和图像分类 (cls)。

## 支持的任务类型

- **det**: 目标检测
- **seg**: 语义分割
- **pose**: 人体姿态估计
- **obb**: 旋转框目标检测
- **cls**: 图像分类

## 环境要求

- **CUDA**: 11.5
- **TensorRT**: 8.2.4.2
- **CMake**: 3.10+
- **C++ 编译器**: 支持 C++11 及以上

## 编译和运行

### 编译步骤

```bash
cd yolov8_tensorrtx
mkdir build
cmake ..
make -j8
```

### 运行示例

```bash
./yolov8 ../model.engine ../images det ../label.txt
```

其中参数说明：
- `../model.engine`: TensorRT 引擎文件路径
- `../images`: 图像文件夹路径
- `det`: 任务类型 (可选值: det, seg, pose, obb, cls)
- `../label.txt`: 标签文件路径

## 项目结构

```
yolov8_tensorrtx/
├── include/
│   ├── block.h
│   ├── config.h
│   ├── cuda_utils.h
│   ├── logging.h
│   ├── macros.h
│   ├── postprocess.h
│   ├── preprocess.h
│   ├── trt_utils.h
│   ├── types.h
│   ├── yololayer.h
│   └── yolov8_trt.h
├── src/
│   ├── block.cpp
│   ├── postprocess.cpp
│   ├── postprocess.cu
│   ├── preprocess.cu
│   ├── yololayer.cu
│   └── yolov8_trt.cpp
├── CMakeLists.txt
├── demo.cpp
└── README.md
```

## 核心功能

- **统一接口**: 将多种任务类型集成到一个类中，提供一致的使用方式
- **高效推理**: 利用 TensorRT 进行模型优化，提高推理速度
- **CUDA 加速**: 使用 CUDA 进行预处理和后处理，进一步提升性能
- **灵活配置**: 支持不同的任务类型和模型配置

## 参考链接

- [tensorrtx](https://github.com/wang-xinyu/tensorrtx): TensorRT 模型部署工具

## 注意事项

- 确保已正确安装 CUDA 和 TensorRT 8
- 模型引擎文件需要通过 tensorrtx 项目生成
- 运行前请确保图像文件夹和标签文件存在

## 测试版本

- CUDA: 11.5
- TensorRT: 8.2.4.2
