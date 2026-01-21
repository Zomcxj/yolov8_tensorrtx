#include "yolov8_trt.h"
int kInputH = 640;
int kInputW = 640;

void YOLOV8_TRTx::setInputSize(int h, int w) {
    kInputH = h;
    kInputW = w;
}

void YOLOV8_TRTx::batch_preprocess(std::vector<cv::Mat>& imgs) {
    for (size_t b = 0; b < imgs.size(); b++) {
        int h = imgs[b].rows;
        int w = imgs[b].cols;
        int m = std::min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        cv::Mat img = imgs[b](cv::Rect(left, top, m, m));
        cv::resize(img, img, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1/255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);

        // CHW format
        for (int c = 0; c < 3; ++c) {
            int i = 0;
            for (int row = 0; row < kInputH; ++row) {
                for (int col = 0; col < kInputW; ++col) {
                cpu_input_buffer[b * 3 * kInputH * kInputW + c * kInputH * kInputW + i] =
                    channels[c].at<float>(row, col);
                ++i;
                }
            }
        }
    }
}

std::vector<float> YOLOV8_TRTx::softmax(float *prob, int n) {
    std::vector<float> res;
    float sum = 0.0f;
    float t;
    for (int i = 0; i < n; i++) {
        t = expf(prob[i]);
        res.push_back(t);
        sum += t;
    }
    for (int i = 0; i < n; i++) {
        res[i] /= sum;
    }
    return res;
}

std::vector<int> YOLOV8_TRTx::topk(const std::vector<float>& vec, int k) {
    std::vector<int> topk_index;
    std::vector<size_t> vec_index(vec.size());
    std::iota(vec_index.begin(), vec_index.end(), 0);

    std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2) { return vec[index_1] > vec[index_2]; });

    int k_num = std::min<int>(vec.size(), k);

    for (int i = 0; i < k_num; ++i) {
        topk_index.push_back(vec_index[i]);
    }

    return topk_index;
}

cv::Rect YOLOV8_TRTx::get_downscale_rect(float bbox[4], float scale) {

    float left = bbox[0];
    float top = bbox[1];
    float right = bbox[0] + bbox[2];
    float bottom = bbox[1] + bbox[3];

    left = left < 0 ? 0 : left;
    top = top < 0 ? 0 : top;
    right = right > kInputW ? kInputW : right;
    bottom = bottom > kInputH ? kInputH : bottom;

    left /= scale;
    top /= scale;
    right /= scale;
    bottom /= scale;
    return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
}

std::vector<cv::Mat> YOLOV8_TRTx::process_mask(const float* proto, int proto_size, std::vector<Detection>& dets) {

    std::vector<cv::Mat> masks;
    for (size_t i = 0; i < dets.size(); i++) {

        cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
        auto r = get_downscale_rect(dets[i].bbox, 4);

        for (int x = r.x; x < r.x + r.width; x++) {
            for (int y = r.y; y < r.y + r.height; y++) {
                float e = 0.0f;
                for (int j = 0; j < 32; j++) {
                    e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
                }
                e = 1.0f / (1.0f + expf(-e));
                mask_mat.at<float>(y, x) = e;
            }
        }
        cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
        masks.push_back(mask_mat);
    }
    return masks;
}

std::vector<std::string> YOLOV8_TRTx::read_classes(std::string file_name) {
    std::vector<std::string> classes;
    std::ifstream ifs(file_name, std::ios::in);
    if (!ifs.is_open()) {
        std::cerr << file_name << " is not found, pls refer to README and download it." << std::endl;
        assert(0);
    }
    std::string s;
    while (std::getline(ifs, s)) {
        classes.push_back(s);
    }
    ifs.close();
    return classes;
}

void YOLOV8_TRTx::deserialize_engine(std::string& engine_name) {
    std::ifstream file(engine_name, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open engine file: " + engine_name);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::unique_ptr<char[]> serialized_engine(new char[size]);
    if (!file.read(serialized_engine.get(), size)) {
        throw std::runtime_error("Failed to read engine data");
    }

    Logger gLogger;
    runtime = createInferRuntime(gLogger);
    assert(runtime);
    engine = runtime->deserializeCudaEngine(serialized_engine.get(), size);
    assert(engine);
    context = engine->createExecutionContext();
    assert(context);
}


void YOLOV8_TRTx::prepare_buffer() {
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // 设备端内存优先分配
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_buffers[0]), kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_buffers[1]), kBatchSize * kOutputSize * sizeof(float)));
    // 主机端内存后分配
    if (cuda_post_process == "c") {
        output_buffer_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            throw std::runtime_error("GPU post-processing not supported for multi-batch yet");
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&decode_ptr_device), sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element))); // 设备端
        decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];      // 主机端后分配
    }
}

void YOLOV8_TRTx::model_infer() {
    auto infer_start = std::chrono::high_resolution_clock::now();
    context->enqueue(kBatchSize, reinterpret_cast<void**>(device_buffers), stream, nullptr);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float), 
                                    cudaMemcpyDeviceToHost, stream));
        if (this->task == "seg") {
            CUDA_CHECK(cudaMemcpyAsync(output_seg_buffer_host, device_buffers[2], kBatchSize * kOutputSegSize * sizeof(float), 
                                        cudaMemcpyDeviceToHost, stream));
        }
        auto infer_end = std::chrono::high_resolution_clock::now();
        std::cout << "Inference time: " << 
                    std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count() << "μs\n";
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode(device_buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), 
                                    cudaMemcpyDeviceToHost, stream));
        auto total_end = std::chrono::high_resolution_clock::now();
        std::cout << "Inference + GPU Post-process time: "  << 
                    std::chrono::duration_cast<std::chrono::microseconds>(total_end - infer_start).count() << "μs\n";
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


void YOLOV8_TRTx::prepare_buffers_cls() {
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_buffers[0]), kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_buffers[1]), kBatchSize * kOutputSize * sizeof(float)));

    cpu_input_buffer = new float[kBatchSize * 3 * kInputH * kInputW];
    output_buffer_host = new float[kBatchSize * kOutputSize];
}

void YOLOV8_TRTx::model_infer_cls() {
    auto start = std::chrono::system_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(device_buffers[0], cpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(kBatchSize, reinterpret_cast<void**>(device_buffers), stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));    
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    cudaStreamSynchronize(stream);
}

void YOLOV8_TRTx::prepare_buffer_seg() {
    assert(engine->getNbBindings() == 3);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    const int outputIndex_seg = engine->getBindingIndex("proto");

    assert(inputIndex == 0);
    assert(outputIndex == 1);
    assert(outputIndex_seg == 2);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_buffers[0]), kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_buffers[1]), kBatchSize * kOutputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_buffers[2]), kBatchSize * kOutputSegSize * sizeof(float)));

    if (cuda_post_process == "c") {
        output_buffer_host = new float[kBatchSize * kOutputSize];
        output_seg_buffer_host = new float[kBatchSize * kOutputSegSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        // Allocate memory for decode_ptr_host and copy to device
        decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&decode_ptr_device), sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}


void YOLOV8_TRTx::init_model(std::string engine_name, std::string labels_txt, std::string task) {

    cudaSetDevice(kGpuId);
    cuda_post_process = is_gpu;
    this->task = task;
    this->class_name = read_classes(labels_txt);

    if (this->task == "seg") {
        read_labels(labels_txt, labels_map);
        // assert(kNumClass == labels_map.size());
        buffer_count = 3;
    } else {
        buffer_count = 2;  
    }

    device_buffers = new float*[buffer_count];
    for (int i = 0; i < buffer_count; ++i) {
        device_buffers[i] = nullptr;
    }
    
    deserialize_engine(engine_name);
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);

    model_bboxes = engine->getBindingDimensions(1).d[0];
    int model_kInputH = engine->getBindingDimensions(0).d[1];
    int model_kInputW = engine->getBindingDimensions(0).d[2];
    setInputSize(model_kInputH, model_kInputW);

    if (this->task == "seg") {
        kOutputSize = kMaxNumOutputBbox * (sizeof(Detection) - sizeof(float) * 51) / sizeof(float) + 1;
        kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);
        prepare_buffer_seg();
    }
    else if (this->task == "cls") {
        kOutputSize = model_bboxes;
        prepare_buffers_cls();    
    }
    else {
        kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
        prepare_buffer();
    }
}

bool YOLOV8_TRTx::trtx_infer(cv::Mat& img, std::vector<TRTxOutput>& result) {

	std::vector<cv::Mat> img_batch = {img};
	std::vector<std::vector<TRTxOutput>> result_batch;
	if (trtx_batch_infer(img_batch, result_batch)) {
		result = result_batch[0];
		return true;
	}
	else 
        return false;
}


bool YOLOV8_TRTx::trtx_batch_infer(std::vector<cv::Mat> &img_batch, std::vector<std::vector<TRTxOutput>> &result_batch) {

    if (this->task == "cls") {
        batch_preprocess(img_batch);
        model_infer_cls();

        // Postprocess and get top-k result
        for (size_t b = 0; b < img_batch.size(); b++) {
            float* p = &output_buffer_host[b * kOutputSize];
            auto res = softmax(p, kOutputSize);
            std::vector<int> topk_idx = topk(res, 2);

            for (auto idx: topk_idx) {
                std::string conf = cv::format("%.2f", res[idx]);
                std::cout << this->class_name[idx] << "  " << conf << std::endl;
            }
            std::vector<TRTxOutput> output;
            TRTxOutput op;
            op.id = topk_idx[0];
            op.confidence = res[0];
            op.name = this->class_name[op.id];
            output.push_back(op);
            result_batch.push_back(output);
        }
        if (result_batch.size())
            return true;
        else
            return false;
    }

    // Preprocess && Run inference
    cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
    model_infer();

    std::vector<std::vector<Detection>> res_batch;
    std::vector<cv::Mat> masks;
    if (this->task == "det" || this->task == "seg") {
        if (cuda_post_process == "c") {
            batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
            if (this->task == "seg") {
                for (size_t b = 0; b < img_batch.size(); b++) {
                    auto& res = res_batch[b];
                    cv::Mat img = img_batch[b];
                    masks = process_mask(&output_seg_buffer_host[b * kOutputSegSize], kOutputSegSize, res);
                }  
            }
        } else if (cuda_post_process == "g") {
            batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
        }
    }
    else if (this->task == "obb") {
        if (cuda_post_process == "c") {
            batch_nms_obb(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
        } else if (cuda_post_process == "g") {
            batch_process_obb(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
        }
    }
    else if (this->task == "pose") {
        if (cuda_post_process == "c") {
            batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
        }
    }

    // Detection transfer TRTxOutput
    for (size_t b = 0; b < img_batch.size(); b++) {
        std::vector<TRTxOutput> output;
        auto& res = res_batch[b];
        for (int i = 0; i < res.size(); i++) {
            if (res[i].conf > 1)
                continue;
            TRTxOutput op;
            op.id = res[i].class_id;
            op.name = this->class_name[op.id];
            op.confidence = res[i].conf;
            if (this->task == "det") {
                op.box = get_rect(img_batch[b], res[i].bbox);
            } 
            else if (this->task == "pose") {
                op.box = get_rect_adapt_landmark(img_batch[b], res[i].bbox, res[i].keypoints);
                for (int k = 0; k < 3 * kNumberOfPoints; k = k + 3) {
                    PosePoint pp;
                    pp.x = res[i].keypoints[k];
                    pp.y = res[i].keypoints[k+1];
                    pp.score = res[i].keypoints[k+2];
                    op.keyPoints.push_back(pp);
                }
            } 
            else if (this->task == "obb") {
                std::vector<cv::Point> corner_points = get_corner(img_batch[b], res[i]);
                op.rotatedBox = cv::minAreaRect(corner_points);
            } 
            else if (this->task == "seg") {
                op.box = get_rect(img_batch[b], res[i].bbox);
                op.boxMask = masks[i];
            }
            output.push_back(op);
        }
        result_batch.push_back(output);
    }
    if (result_batch.size())
        return true;
    else
        return false;

    // draw Detection
    // if (task == "det")
    //     draw_bbox(img_batch, res_batch);
    // else if (task == "pose")
    //     draw_bbox_keypoints_line(img_batch, res_batch);
    // else if (task == "obb")
    //     draw_bbox_obb(img_batch, res_batch);
    // else if (task == "seg")
    //     draw_mask_bbox(img_batch[0], res_batch[0], masks, labels_map); 

    // if (res_batch.size())
    //     return true;
    // else
    //     return false;

}


void YOLOV8_TRTx::destroy() {
    // Release stream and buffers
    cudaStreamDestroy(stream);
    for (int i = 0; i < buffer_count; ++i) {
        if (device_buffers[i] != nullptr) 
            CUDA_CHECK(cudaFree(device_buffers[i]));
    }
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    delete[] output_buffer_host;
    delete[] output_seg_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
}

void YOLOV8_TRTx::draw_result(cv::Mat& img, std::vector<TRTxOutput> result)
{   
	double scale;
    int thickness;
    scale = 0.5, thickness = 1;

    for (int i = 0; i < result.size(); i++)
	{
		if (result[i].confidence < 0.5)
            continue;

        int left, top;
        if (result[i].box.area() > 0)  // HBB
        {
            cv::rectangle(img, result[i].box, cv::Scalar(100, 100, 255), thickness, cv::LINE_AA);
            left = result[i].box.x;
            top = result[i].box.y;
        }
        else if (result[i].rotatedBox.size.area() > 0)  // OBB
        {
            cv::Point2f p[4];
            result[i].rotatedBox.points(p);
            for (int l = 0; l < 4; l++) 
            {
                cv::line(img, p[l], p[(l + 1) % 4], cv::Scalar(100, 100, 255), thickness, cv::LINE_AA);
            }
            left = result[i].rotatedBox.center.x;
            top = result[i].rotatedBox.center.y;
        }

        std::string conf = cv::format("%.2f", result[i].confidence);
        std::string label = result[i].name + ":" + conf;
        int idx = result[i].id + 3;
        cv::Scalar color = cv::Scalar(37 * idx * idx % 255, 17 * idx * idx % 255, 29 * idx * idx % 255);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX, scale, thickness, &baseLine);
        cv::rectangle(img, cv::Point(left, top - labelSize.height - 2), cv::Point(left + labelSize.width, top - 2), color, -1, cv::LINE_AA);
        cv::putText(img, label, cv::Point(left, top - 2), cv::FONT_HERSHEY_COMPLEX, scale, cv::Scalar(225, 255, 255), thickness, cv::LINE_AA);

        // std::cout << result[i].name << " @ " << conf << std::endl;

        if (task == "pose") {
            for (int j = 0; j < result[i].keyPoints.size(); ++j) {
                PosePoint kpt = result[i].keyPoints[j];
                if (kpt.score < 0.5)
                    continue;
                if (j < 5)
                    cv::circle(img, cv::Point(kpt.x, kpt.y), 5, cv::Scalar(0, 0, 255), -1, 8);
                else if (j < 11)
                    cv::circle(img, cv::Point(kpt.x, kpt.y), 5, cv::Scalar(0, 255, 0), -1, 8);
                else
                    cv::circle(img, cv::Point(kpt.x, kpt.y), 5, cv::Scalar(255, 0, 0), -1, 8);
            }
        }
        else if (task == "seg") {
            cv::Mat img_mask = scale_mask(result[i].boxMask, img);    // recover mask size to img
            cv::Rect r = result[i].box;
            // mask color
            for (int x = r.x; x < r.x + r.width; x++) {
                for (int y = r.y; y < r.y + r.height; y++) {
                    float val = img_mask.at<float>(y, x);
                    if (val <= 0.5)
                        continue;
                    img.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[0] / 2 + color[0] / 2;
                    img.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1] / 2 + color[1] / 2;
                    img.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[2] / 2 + color[2] / 2;
                }
            }
        }
	}
}