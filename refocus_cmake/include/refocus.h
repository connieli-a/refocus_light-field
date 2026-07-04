// refocus_cmake.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。


#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h> 
#include <Windows.h>


// TODO: 在此处引用程序需要的其他标头。
using namespace std;
using namespace GenApi; // NodeMap / CFloatPtr / CEnumerationPtr


struct CircleInf
{
    /* data */
    float x;
    float y;
    float radius;
    int valid;
    CircleInf(): x(0.0f), y(0.0f), radius(0.0f), valid(0){}
    CircleInf(float _x, float _y, float r): x(_x), y(_y), radius(r), valid(1) {}
    
};
struct Result
{
    float brenner;
    float z;
};
struct ResultData
{
    std::vector<float> center_img;
    std::vector<float> best_image;
    float best_value;

};
struct FrameBuffer
{
    cv::Mat bufferA, bufferB;
    atomic<cv::Mat*> writeBuf{&bufferA};
    atomic<cv::Mat*> readBuf{&bufferB};
    atomic<bool> newFrame{false};
    mutex mtx;
    condition_variable cv;
};
struct ResultBuffer
{
    ResultData bufA;
    ResultData bufB;
    std::atomic<ResultData*> writeBuf{&bufA};
    std::atomic<ResultData*> readBuf{&bufB};
    // std::pair<vector<float>, float> bufA, bufB;
    // atomic<std::pair<vector<float>, float>*> writeBuf{&bufA};
    // atomic<std::pair<vector<float>, float>*> readBuf{&bufB};
    atomic<bool> newResult{false};
    mutex mtx;
    condition_variable cv;
};
struct DisplayBuffer {
    std::mutex mtx;
    cv::Mat img;
    cv::Mat img_large;
    cv::Mat img_center;//midcenter_subaperture image
    bool hasNew = false;
} ;
struct CameraDisplayBuffer {
    std::mutex mtx;
    cv::Mat fullframe;
    bool hasNew = false;
};
class ImageProcessor {
    public:
    virtual ~ImageProcessor() = default;
    
    static std::shared_ptr<ImageProcessor> create(const vector<CircleInf>& circleList, const float y_tolerance, const int patch_size_cpp, const int image_rows, const int image_cols, const int32_t device = 0, const bool useGraphe = true);

    
    // pure virtual functions, only defining interfaces
    virtual void imageprocess_cuda(
    const cv::Mat& image_mla) = 0;         // CV_8UC3
    virtual void show_image(const ResultData& result_frame) = 0;  
                  
    // virtual vector<cv::Vec3f> currentimage() = 0;
    
    virtual int get_col() const = 0;
    virtual int get_row() const = 0;
    
    protected:
    int n_cols;
    int n_rows;
    
};

extern FrameBuffer frameBuffer;
extern ResultBuffer resultBuffer;
extern std::atomic<bool> running;

extern DisplayBuffer displayBuffer;
extern CameraDisplayBuffer cameraDisplayBuffer;
extern int rangex1;
extern int rangex2;
extern int rangey1;
extern int rangey2;
extern cv::Mat image_mla_roi;
void gpuThread(std::shared_ptr<ImageProcessor> refocus_pointer);
void showThread(std::shared_ptr<ImageProcessor> refocus_pointer);
void cameraThread();
std::pair<float, float> get2d_coordinate(const cv::Mat& center_image, int black_threshold);







