#pragma once
#include <map>
#include <string>
#include <vector>
#include "NvInfer.h"
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "config.h"
#include "yololayer.h"

int calculateP(int ksize);

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

nvinfer1::ILayer* convBnSiLU(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                        int ch, int k, int s, int p, std::string lname);

nvinfer1::ILayer* C2F(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                 int c2, int n, bool shortcut, float e, std::string lname);

nvinfer1::ILayer* C2(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c1,
                                int c2, int n, bool shortcut, float e, std::string lname);

nvinfer1::ILayer* C3(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                int c2, int n, bool shortcut, float e, std::string lname);

nvinfer1::ILayer* SPPF(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                  int c2, int k, std::string lname);

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int ch, int grid, int k, int s, int p, std::string lname);

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
                                       std::vector<nvinfer1::IConcatenationLayer*> dets, const int* px_arry,
                                       int px_arry_num, int num_class, bool is_segmentation, bool is_pose, bool is_obb);
