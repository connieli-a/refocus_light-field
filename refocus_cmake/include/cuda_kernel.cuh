#ifndef __CUDACC__
#  define __host__
#  define __device__
#endif

#pragma once
#include "refocus.h"
#include <cuda_runtime.h>
#include <cuda.h>


struct Vec3f {
    float x, y, z;

    __host__ __device__ Vec3f() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}

    __host__ __device__ float& operator[](int i) { return ((&x)[i]); }
    __host__ __device__ const float& operator[](int i) const { return ((&x)[i]); }
};
// 继承自父类，负责 CUDA 部分
class GPUProcessor : public ImageProcessor {
public:
    GPUProcessor(const vector<CircleInf>& circleList, const float y_tolerance, const int patch_size,  const int image_rows, const int image_cols);
    virtual ~GPUProcessor();

    void imageprocess_cuda(
    const cv::Mat& image_mla) override;                    // CV_8UC3
    // vector<cv::Vec3f> currentimage() override;

    void show_image(std::pair<vector<float>, float> result_frame) override;    
    int get_col() const override { return n_cols; }
    int get_row() const override { return n_rows; }
private: 
    void cuda_preprocess( const cv::Mat& image_mla);
    void prepare_data();
    void extract_rows(const vector<CircleInf>& circleList, const float y_tolerance);
    // void preprocess(const vector<cv::Vec3f>& circles, vector<CircleInf>& sortedList);
    
   
    // float find_bestvalue(float z0, const float& fine_step);
    // float Equation_solving( vector<float>& y, const vector<float>& x);
    float quadratic_fit_points(const std::vector<float>& y, const std::vector<float>& x);
    // float parabolic_peak_refine(const vector<float>& brenner_scores, const vector<float>& depth);
    void generate_best_view(const float& z_best, float& score);
    
    //--------related CUDA
    
    // int32_t m_device ;
    // bool m_useGraph;
    
    cudaStream_t m_stream ;
    // cudaGraphExec_t m_graphExec;

    CircleInf* d_rows_flat = nullptr;
    int* d_rows_offsets = nullptr;
   
    float* d_imagefloat = nullptr;//transform_type
    float* d_images = nullptr;//save the data after transform
    float* d_volume = nullptr;
  
  
    float* d_brenner_scores = nullptr;
    float* d_alpha = nullptr;//depths of the num_plane
    float* d_volume_1 = nullptr;//generate display image
    float* d_alpha_1 = nullptr;
    // Vec3f* d_image_1 = nullptr;
    // Vec3f* d_volume_n = nullptr;//estimated algorithm
    // float* d_brenner_scores_n = nullptr;
    // float* d_alpha_n = nullptr;  

    int image_rows;
    int image_cols;
    // vector<float>& depth_range;
    cv::cuda::GpuMat d_img;
    //---------the parament of cpu logic
    vector<vector<CircleInf>> rows;

    std::vector<CircleInf> rows_flat;             // 已经 prepare_data 过
    std::vector<int> rows_offsets;                // 已经 prepare_data 过
    int total_circles ;
    int total_uv   = n_cols * n_rows;
    int total_pix  = patch_area * total_uv;
    //---optical parament

    int patch_size = 0;
    int patch_area = patch_size * patch_size;
    int num_depth_plane = 5;
    int mid_idx = patch_size / 2;
    // float pixel_size = 2.0;
    // int s = 125;//the diameter of the lens micrometer
    // int f = 2500;// the focal length of the lens micrometer
    // int num_plane = 5;
    vector<float> alpha_range;
    float start = 0.85;
    float end = 1.25;
    float step = static_cast<float>(end - start) / static_cast<float>(num_depth_plane - 1);
  
    
};
