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
    GPUProcessor(const vector<cv::Vec3f>& circles, const float y_tolerance, const int patch_size, const int num_depth_plane, const std::vector<float> disparity_x_flat, const std::vector<float> disparity_y_flat);
    virtual ~GPUProcessor();

    std::vector<cv::Vec3f> imageprocess_cuda(
    const cv::Mat& image_mla ) override;                    // CV_8UC3
    

   
   
    //  // 声明 kernel
    // __device__ inline Vec3f bilinear_lookup(const Vec3f* img, int n_rows, int n_cols, float y, float x);
    // __global__ void transform_kernel( const Vec3f* __restrict__ input_image, int image_rows, int image_cols,
    // const CircleInf* __restrict__ rows_flat, const int* __restrict__ rows_offsets,
    // int n_rows, int n_cols, int total_circles,
    // int patch_size, float half,
    // Vec3f* __restrict__ out);
    // __global__ void shift_and_sum_kernel(const Vec3f* d_images,  // [n_rows][n_cols][patch_area]
    //                       Vec3f* d_volume,        // [num_depth_plane][n_rows][n_cols]
    //                       const float* d_disp_x,  // [patch_area][num_depth_plane]
    //                       const float* d_disp_y,  // [patch_area][num_depth_plane]
    //                       int n_rows, int n_cols,
    //                       int patch_size, int num_depth_plane);    

    // __global__ void unchar_to_vec3f_kernel(const uchar3* __restrict__ src, Vec3f* __restrict__ dst, int rows, int cols);

    int get_col() const override { return n_cols; }
    int get_row() const override { return n_rows; }
private: 
    void cuda_preprocess( const cv::Mat& image_mla);
    void prepare_data();
    void extract_rows(const vector<cv::Vec3f>& circles, const float y_tolerance);
    void preprocess(const vector<cv::Vec3f>& circles, vector<CircleInf>& sortedList);
    //--------related CUDA
    
    // int32_t m_device ;
    // bool m_useGraph;
    
    cudaStream_t m_stream ;
    // cudaGraphExec_t m_graphExec;

    CircleInf* d_rows_flat = nullptr;
    int* d_rows_offsets = nullptr;
    float* d_disp_x = nullptr;
    float* d_disp_y = nullptr;
    Vec3f* d_imagefloat ;
    Vec3f* d_images ;
    Vec3f* d_volume = nullptr;

    int image_rows;
    int image_cols;
    //---------the parament of cpu logic
    vector<vector<CircleInf>> rows;

    std::vector<CircleInf> rows_flat;             // 已经 prepare_data 过
    std::vector<int> rows_offsets;                // 已经 prepare_data 过
    int total_circles ;
    int total_uv   = n_cols * n_rows;
    int total_pix  = patch_area * total_uv;
   std::vector<float> disparity_x_flat;
    std::vector<float> disparity_y_flat;
    //---optical parament

    int patch_size = 0;
    int patch_area = patch_size * patch_size;
    int num_depth_plane;
};
