#pragma once
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "postprocess.h"
#include "preprocess.h"
#include "trt_utils.h"
#include <chrono>
#include <cmath>
#include <numeric>
using namespace nvinfer1;    

struct PosePoint {
	float x = 0;
	float y = 0;
	float score = 0;
};

struct TRTxOutput {
	int id;             
	float confidence;  
	cv::Rect box;                   // hbb结果矩形框
    cv::RotatedRect rotatedBox;     // obb结果矩形框
	cv::Mat boxMask;                // box掩码
	std::vector<PosePoint> keyPoints; // 位姿
    std::string name;
};


class YOLOV8_TRTx
{
public:   

    void init_model(std::string engine_name, std::string labels_txt, std::string task);
    bool trtx_infer(cv::Mat& img, std::vector<TRTxOutput>& result);
    bool trtx_batch_infer(std::vector<cv::Mat> &img_batch, std::vector<std::vector<TRTxOutput>> &result_batch);
    void destroy();
    void draw_result(cv::Mat& img, std::vector<TRTxOutput> result);

    float kNmsThresh = 0.5f;
    float kConfThresh = 0.5f;

private:

    std::string cuda_post_process;
    std::string task;
    std::string is_gpu = "c";
    int kOutputSize;
    int kOutputSegSize;
    int model_bboxes;  // output_dims 
    std::vector<std::string> class_name;
    std::unordered_map<int, std::string> labels_map;
    
    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream;

    // Prepare cpu and gpu buffers
    int buffer_count;
    float** device_buffers;
    float* cpu_input_buffer = nullptr;
    float* output_buffer_host = nullptr;
    float* output_seg_buffer_host = nullptr;
    float* decode_ptr_host = nullptr;
    float* decode_ptr_device = nullptr;

    void setInputSize(int h, int w);
    void batch_preprocess(std::vector<cv::Mat>& imgs);
    std::vector<float> softmax(float *prob, int n);
    std::vector<int> topk(const std::vector<float>& vec, int k);
    std::vector<std::string> read_classes(std::string file_name);

    cv::Rect get_downscale_rect(float bbox[4], float scale);
    std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets);

    void deserialize_engine(std::string& engine_name);
    void prepare_buffer();
    void model_infer();

    void prepare_buffers_cls();
    void model_infer_cls();
    
    void prepare_buffer_seg();
};