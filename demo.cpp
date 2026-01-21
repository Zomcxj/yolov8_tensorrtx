#include "yolov8_trt.h"

int main(int argc, char** argv){

    std::string engine_name = argv[1];
    std::string img_dir = argv[2];
    std::string task = argv[3];    // det seg pose obb cls
    std::string labels_txt = argv[4];

    auto start = std::chrono::system_clock::now();

    YOLOV8_TRTx trt;
    trt.init_model(engine_name, labels_txt, task);

    std::vector<std::string> fn;
    cv::glob(img_dir, fn, false);
    int loop_time = 1;
    for (int i = 0; i < loop_time; i++) 
    {
        for (int j = 0; j < fn.size(); j++) 
        {
            cv::Mat img = cv::imread(fn[j]);
            std::string::size_type iPos = fn[j].find_last_of('/') + 1;
            std::string filename = fn[j].substr(iPos, fn[j].length() - iPos);
            // std::cout << "inference pic " << filename << std::endl;

            // infer
            std::vector<TRTxOutput> result;
            trt.kNmsThresh = 0.8f;
            trt.trtx_infer(img, result);

            // draw&&save
            if (result.size())
                trt.draw_result(img, result);
            if (task != "cls")
                cv::imwrite("../outputs/" + filename, img);
        }
    }
    trt.destroy();

    auto end = std::chrono::system_clock::now();
    int sum_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    int average_time = sum_time / fn.size() / loop_time;
    std::cout << "sum time: " << sum_time << "ms" << std::endl;
    std::cout << "file: " << fn.size() * loop_time << " average time: " << average_time << "ms" << std::endl;
    return 0;
}
