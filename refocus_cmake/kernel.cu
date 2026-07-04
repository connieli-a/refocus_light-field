#include "include/cuda_kernel.cuh"
#include "include/DAControl.h"
#include "include/ADControl.h"
#include "include/refocus.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>


// static std::ofstream csv("brenner_curve.csv", std::ios::out | std::ios::app);
// static bool csv_header_written = false;

GPUProcessor::GPUProcessor(const vector<CircleInf>& circleList, const float y_tolerance, const int patch_size_cpp,  const int image_rows, const int image_cols):  patch_size(patch_size_cpp),   image_rows(image_rows), image_cols(image_cols){
    extract_rows(circleList, y_tolerance);
    prepare_data();
    if(num_depth_plane == 1)
        alpha_range.push_back(1.0);
    else{
        for (int i = 0; i < num_depth_plane; ++i)
        {
            alpha_range.push_back(start + step * i);
        }
    }
    //set the GPU information 
    // 使用GPUの設定
    cudaStreamCreate(&m_stream);

    cudaMalloc(&d_rows_flat, sizeof(CircleInf) * total_circles);
    cudaMalloc(&d_rows_offsets, sizeof(int) * rows_offsets.size());
    cudaMalloc(&d_imagefloat, sizeof(float) * image_rows * image_cols);
    cudaMalloc(&d_images, sizeof(float) * n_rows * n_cols * patch_area);
    cudaMalloc(&d_volume, num_depth_plane * n_rows * n_cols * sizeof(float));
    cudaMalloc(&d_alpha, sizeof(float) * num_depth_plane);
    cudaMalloc(&d_brenner_scores, sizeof(float) * num_depth_plane);
    cudaMalloc(&d_volume_1, n_rows * n_cols * sizeof(float));
    cudaMalloc(&d_alpha_1, sizeof(float) );
    // cudaMalloc(&d_image_1, sizeof(Vec3f) * n_rows * n_cols);
    // cudaMalloc(&d_volume_n, num_plane * n_rows * n_cols * sizeof(Vec3f));
    // cudaMalloc(&d_alpha_n, sizeof(float) * num_plane);
    // cudaMalloc(&d_brenner_scores_n, sizeof(float) * num_plane);
    // cudaMemcpy(d_rows_flat, rows_flat.data(),
    // sizeof(CircleInf) * total_circles, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_rows_offsets, rows_offsets.data(),
    // sizeof(int) * rows_offsets.size(), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_rows_flat, rows_flat.data(),
                sizeof(CircleInf) * total_circles,
                cudaMemcpyHostToDevice, m_stream);

    cudaMemcpyAsync(d_rows_offsets, rows_offsets.data(),
                sizeof(int) * rows_offsets.size(),
                cudaMemcpyHostToDevice, m_stream);

    cudaMemcpyAsync(d_alpha, alpha_range.data(),
                sizeof(float) * num_depth_plane,
                cudaMemcpyHostToDevice, m_stream);
    d_img.create(cv::Size(image_cols, image_rows), CV_8UC1);
}
GPUProcessor::~GPUProcessor(){

    cudaStreamSynchronize(m_stream);
    cudaStreamDestroy(m_stream);

    cudaFree(d_rows_flat);
    cudaFree(d_rows_offsets);
    cudaFree(d_imagefloat);
    cudaFree(d_images);
    cudaFree(d_volume);
    cudaFree(d_alpha);     
    cudaFree(d_brenner_scores);
    cudaFree(d_volume_1);
    cudaFree(d_alpha_1);
    // cudaFree(d_image_1);
    // cudaFree(d_volume_n);
    // cudaFree(d_alpha_n);     
    // cudaFree(d_brenner_scores_n);
}

//类型转换 type conversion
__global__ void copy_kernel(const uchar* __restrict__ src, int rows, int cols, size_t step, 
                            float* __restrict__ dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    const uchar* row_ptr = (const uchar*)((const char*)src + y * step);
    uchar p = row_ptr[x];

    int idx = y * cols + x;
    // int idx = y * cols + x;
    // uchar3 p = src[idx]; 
    dst[idx] = float(p)/255.f;
}
//双线性插值bilinear interpolation
__device__ inline float bilinear_lookup(
    const float* img, int n_rows, int n_cols, float x, float y)
{
    int ix = floorf(x);
    int iy = floorf(y);
    float dx = x - ix;
    float dy = y - iy;

    if (ix < 0 || ix + 1 >= n_cols || iy < 0 || iy + 1 >= n_rows)
        return 0;

    float c00 = img[iy * n_cols + ix];
    float c01 = img[iy * n_cols + (ix + 1)];
    float c10 = img[(iy + 1) * n_cols + ix];
    float c11 = img[(iy + 1) * n_cols + (ix + 1)];

    float val = (1 - dx) * (1 - dy) * c00 + dx * (1 - dy) * c01
          + (1 - dx) * dy * c10 + dx * dy * c11;
   
    return val;
}

__global__ void transform_kernel( float* d_img, int image_rows, int image_cols, 
    const CircleInf* __restrict__ rows_flat, const int* __restrict__ rows_offsets,
    int total_circles, int n_rows, int n_cols, int patch_size,
    float* __restrict__ out){
    if(!d_img ||!rows_flat||!rows_offsets ) return; 
    int patch_area = patch_size * patch_size;
    int total = n_rows * n_cols * patch_area;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    
   
    // if(tid == 0){
    //     printf("kernel running!\n");
    // }
    
    float half = (patch_size - 1) * 0.5f;        
    // 还原 (uv, s, t)
    int uv_idx = tid % (n_rows*n_cols); // 每 patch 内 uv_idx
    int s = (tid / (n_rows*n_cols)) / patch_size;
    int t = (tid / (n_rows*n_cols)) % patch_size;

    int u = uv_idx / n_cols;
    int v = uv_idx % n_cols;
    
    
    // 这一行的起止（可处理“每行列数不同”的情况）
    int start = rows_offsets[u];
    // int end   = (u + 1 < n_rows) ? rows_offsets[u + 1] : total_circles;
    // int row_len = end - start;

    // // 若该 (u,v) 在这一行不存在，直接填零
    // if (v >= row_len) {
    //     out[3 * tid + 0] = 0.f;
    //     out[3 * tid + 1] = 0.f;
    //     out[3 * tid + 2] = 0.f;
    //     return;
    // }

    CircleInf ci = rows_flat[start + v];
    int idx_out = (s * patch_size + t) * n_rows * n_cols + u * n_cols + v;
    if (!ci.valid) {
      out[idx_out] = 0;
        return;
    }

    // 计算取样位置（与 CPU 代码一致）
    float x = ci.x + (t - half);
    float y = ci.y + (s - half);

    float bgr = bilinear_lookup(d_img, image_rows, image_cols,  x, y);
    out[idx_out] = bgr;
}

// __global__ void transform_kernel_centerview( Vec3f* d_img, int image_rows, int image_cols, 
//     const CircleInf* __restrict__ rows_flat, const int* __restrict__ rows_offsets,
//     int total_circles, int n_rows, int n_cols, int patch_size,
//     Vec3f* __restrict__ out){
//     if(!d_img ||!rows_flat||!rows_offsets ) return; 
    
    
//     int uv_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (uv_idx >= n_rows * n_cols) return;
   
//     float half = (patch_size - 1) * 0.5f;        
//     // 还原 (uv, s, t)
   
//     int s = half;
//     int t = half;

//     int u = uv_idx / n_cols;
//     int v = uv_idx % n_cols;
    
    
//     // 这一行的起止（可处理“每行列数不同”的情况）
//     int start = rows_offsets[u];
   
//     CircleInf ci = rows_flat[start + v];
//     if (!ci.valid) {
//       out[u * n_cols + v] = Vec3f(0,0,0);
//         return;
//     }

//     // 计算取样位置（与 CPU 代码一致）
//     float x = ci.x + (t - half);
//     float y = ci.y + (s - half);

//     Vec3f bgr = bilinear_lookup(d_img, image_rows, image_cols,  x, y);
//     out[u * n_cols + v] = bgr;
// }


// __global__ void shift_and_sum_kernel(const Vec3f* d_images,  // [patch_area][n_rows][n_cols]
//                           Vec3f* d_volume,        // [num_depth_plane][n_rows][n_cols]
//                           int n_rows, int n_cols, int patch_size,
//                           const float* d_depth, const int num_depth_plane, float pixel_size, int s,//the diameter of the lens
//                             int f )//the focal length
// {
//     int u = blockIdx.y * blockDim.y + threadIdx.y;//row
//     int v = blockIdx.x * blockDim.x + threadIdx.x;//col
//     int z = blockIdx.z;  // 当前深度平面

//     if (u >= n_rows || v >= n_cols || z >= num_depth_plane) return;
//     int mid_idx = (patch_size - 1) * 0.5f;
//     int patch_area = patch_size * patch_size;
//     Vec3f sum = {0, 0, 0};
//     float depth = d_depth[z];
//     float factor = s * (depth / ((float)f + depth)) / pixel_size;
//     float radius = mid_idx * 0.75f ;
//     int count = 0;
//     for (int idx = 0; idx < patch_area; idx++) {
       
//         int h = idx / patch_size;
//         int w = idx % patch_size;
//         float r2 = (h - mid_idx) * (h - mid_idx) + (w - mid_idx) * (w - mid_idx);
//         if(r2 <= radius * radius){
//             // disparity compute
//             count++;
//             float dx = (w - mid_idx) * factor;
//             float dy = (h - mid_idx) * factor;

//             // 原始像素位置（一个一个views检查）
//             int img_base_idx = idx * n_rows * n_cols;

//             // 浮点位移，使用双线性插值
//             float uu = (float)u + dy;
//             float vv = (float)v + dx;
//             // uu = fminf(fmaxf(uu, 0.f), n_rows - 1.001f);
//             // vv = fminf(fmaxf(vv, 0.f), n_cols - 1.001f);
//             Vec3f val = bilinear_lookup(d_images + img_base_idx, n_rows, n_cols, vv, uu);

//             sum[0] += val[0];
//             sum[1] += val[1];
//             sum[2] += val[2];
//         }
//     }

//     // 平均
//     sum[0] /= count;
//     sum[1] /= count;
//     sum[2] /= count;

    
//     // 输出到 refocused volume
//     int out_idx = z * n_rows * n_cols + u * n_cols + v;
//     d_volume[out_idx] = sum;

// }
__global__ void shift_and_sum_kernel(const float* d_images,  // [patch_area][n_rows][n_cols]
                          float* d_volume,        // [num_depth_plane][n_rows][n_cols]
                          int n_rows, int n_cols, int patch_size,
                          const float* d_alpha, const int num_depth_plane)
{
    int u = blockIdx.y * blockDim.y + threadIdx.y;//row
    int v = blockIdx.x * blockDim.x + threadIdx.x;//col
    int z = blockIdx.z;  // 当前深度平面

    if (u >= n_rows || v >= n_cols || z >= num_depth_plane) return;
    int mid_idx = (patch_size - 1) * 0.5f;
    int patch_area = patch_size * patch_size;
    float sum = 0;
    float alpha = d_alpha[z];
    float factor;
    if(alpha == 0)
        factor = -1;
    else
        factor = 1 - 1 / alpha;
    float radius = mid_idx * 0.75f ;
    int count = 0;
    for (int idx = 0; idx < patch_area; idx++) {
       
        int h = idx / patch_size;
        int w = idx % patch_size;
        float r2 = (h - mid_idx) * (h - mid_idx) + (w - mid_idx) * (w - mid_idx);
        if(r2 <= radius * radius){
            // disparity compute
            count++;
            float dx = (w - mid_idx) * factor;
            float dy = (h - mid_idx) * factor;

            // 原始像素位置（一个一个views检查）
            int img_base_idx = idx * n_rows * n_cols;

            // 浮点位移，使用双线性插值
            float uu = (float)u + dy;
            float vv = (float)v + dx;
            uu = fminf(fmaxf(uu, 0.f), n_rows - 1.001f);
            vv = fminf(fmaxf(vv, 0.f), n_cols - 1.001f);
            float val = bilinear_lookup(d_images + img_base_idx, n_rows, n_cols, vv, uu);

            sum += val;
        }
    }

    // 平均
    if(count > 0)
        sum /= count;
    
    // 输出到 refocused volume
    int out_idx = z * n_rows * n_cols + u * n_cols + v;
    d_volume[out_idx] = sum;

}
__global__ void brenner_kernel(const float* d_volume, float* d_brenner_scores, int m, int n_rows, int n_cols, const int num_depth_plane){
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z;  // 当前深度平面

    if (u >= n_rows || v >= n_cols || z >= num_depth_plane) return;
    float brenner_val = 0.0f;
    int idx = z * n_rows * n_cols + u * n_cols + v;
    float center = d_volume[idx];
    
    float bu = 0.00f;
    float bv = 0.00f;
    if(u + m < n_rows){
        float down = d_volume[z * n_rows * n_cols + (u + m) * n_cols + v];
        bu += (center - down) * (center - down) ;
    }
    if(u - m >= 0) {
        float up = d_volume[z * n_rows * n_cols + (u - m) * n_cols + v];
        bu += (center - up) * (center - up);
    }
    if(v + m < n_cols){
        float right = d_volume[z * n_rows * n_cols + u  * n_cols + (v + m)];
        bv += (center - right) * (center - right);
    }
     if(v - m >= 0) {
        float left = d_volume[z * n_rows * n_cols + u * n_cols + (v - m)];
        bv += (center - left) * (center - left);
    }
    brenner_val = fmaxf(bu, bv);
    atomicAdd(&d_brenner_scores[z], brenner_val);
   
}


void GPUProcessor::imageprocess_cuda(
    const cv::Mat& image_mla               // CV_8UC3      
){
    
    
      //------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    //-------------------------
    cudaEventRecord(start);
    CV_Assert(image_mla.type() == CV_8UC1);
    CV_Assert(patch_size > 0);
    
    //  配置 kernel 维度并 launch
    int threads = 256;
    int total_threads = n_rows * n_cols * patch_area;
    int transform_blocks = (total_threads + threads - 1) / threads;
    
   
    cuda_preprocess(image_mla);
  
    // // 拷回 CPU
    // std::vector<float> h_images(image_rows * image_cols);
    // cudaMemcpy(h_images.data(), d_imagefloat, h_images.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // cv::Mat img(image_rows, image_cols, CV_32FC1, h_images.data());
    
    // cv::Mat img8;
    // img.convertTo(img8, CV_8UC1,255.0);
    // cv::imwrite("orgin.jpg", img8);
    // cudaEventRecord(start);
    

    // 在同一 stream 里执行 kernel
    transform_kernel<<<transform_blocks, threads, 0, m_stream>>>(
        d_imagefloat, image_rows, image_cols, 
        d_rows_flat, d_rows_offsets,
        total_circles, n_rows, n_cols, patch_size,
        d_images
    );
    cudaError_t err_transform = cudaGetLastError();
    if(err_transform != cudaSuccess) printf("transform_Kernel error: %s\n", cudaGetErrorString(err_transform));
    cudaDeviceSynchronize();
   


    // if(!center_img.empty()){
    //     cv::imshow("center image", center_show);
    //     cv::waitKey(1);
    // }else
    //     std::cout<<"center_show is empty"<<std::endl;
    // std::vector<float> h_images(n_rows * n_cols * patch_area );
    // cudaMemcpy(h_images.data(), d_images, h_images.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // std::string folder = "transform_image";
    // if(!fs::exists(folder)){
    //     fs::create_directory(folder);  
    // }
    // for(int s = 0; s < patch_size ; s++){
    //     for(int t = 0; t < patch_size; t++){
    //         int offset = (s * patch_size + t) * (n_rows * n_cols);
    //         cv::Mat img(n_rows, n_cols, CV_32FC1, h_images.data()  + offset);
            
    //         cv::Mat img8;
    //         img.convertTo(img8, CV_8UC1,255.0);
           
            
            
    //         std::string filename = folder + "/output"+ std::to_string(s*patch_size +t) +".png";
    //         cv::imwrite(filename, img8);
        
    //     }
    // }
    // kernel 配置


    dim3 block(16, 16);
    dim3 grid((n_cols+15)/16, (n_rows+15)/16, num_depth_plane);
    
    //generate depth_range
    // cudaMemcpy(d_depth, depth_range.data(),
    // sizeof(float) * num_depth_plane, cudaMemcpyHostToDevice);
   
    
    shift_and_sum_kernel<<<grid, block, 0, m_stream>>>(d_images, d_volume,
        n_rows, n_cols, patch_size, 
        d_alpha, num_depth_plane);
    cudaError_t err_shift = cudaGetLastError();
    if(err_shift != cudaSuccess) printf("shift_Kernel error: %s\n", cudaGetErrorString(err_shift));
    cudaDeviceSynchronize();
    
    // std::vector<float> h_volume(num_depth_plane * n_rows * n_cols);
    // cudaMemcpy(h_volume.data(), d_volume,
    // h_volume.size() * sizeof(float),
    // cudaMemcpyDeviceToHost);
    // std::string folder = "shift_image";
    // if(!fs::exists(folder)){
    //     fs::create_directory(folder);  
    // }
        
    
 
    // for(int slice_idx = 0; slice_idx < num_depth_plane; slice_idx++){
    //     int offset = slice_idx * (n_rows * n_cols);
    //     cv::Mat img(n_rows, n_cols, CV_32FC1, h_volume.data()  + offset);
    //     cv::Mat img8, img8_large;
    //     img.convertTo(img8, CV_8UC1, 255.0);
    //     cv::resize(img8, img8_large, cv::Size(), 10.0 ,10.0,cv::INTER_NEAREST);
    //     std::string filename = folder + "/output"+ std::to_string(slice_idx) +".png";
    //     cv::imwrite(filename, img8_large);
    
    // }       
    int m = 2;
    cudaMemsetAsync(d_brenner_scores, 0, sizeof(float) * num_depth_plane, m_stream);
    brenner_kernel<<<grid, block, 0, m_stream>>>(d_volume, d_brenner_scores, m, n_rows, n_cols, num_depth_plane);
    cudaError_t err_brenner = cudaGetLastError();
    if(err_brenner != cudaSuccess) printf("brenner_Kernel error: %s\n", cudaGetErrorString(err_brenner));
    
    
        
    std::vector<float> h_brenner_scores(num_depth_plane);
    // cudaMemcpy(h_brenner_scores.data(), d_brenner_scores,
    // h_brenner_scores.size() * sizeof(float),
    // cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(h_brenner_scores.data(), d_brenner_scores,
                sizeof(float) * num_depth_plane,
                cudaMemcpyDeviceToHost, m_stream);
    cudaDeviceSynchronize();


   
   
    auto it = std::max_element(h_brenner_scores.begin(), h_brenner_scores.end());
    int max_idx = std::distance(h_brenner_scores.begin(), it);
    float z_coarse = alpha_range[max_idx];
    float max_score = *it;
    float min_score = *std::min_element(h_brenner_scores.begin(), h_brenner_scores.end());
    // cout<<"coarse_best:"<<z_coarse<<", score = "<<max_score<<endl;
    float peak_contract = (max_score - min_score) / max_score;
    float z_best = z_coarse;
    // cout<<"peak_contract:"<<peak_contract<<endl;
    
    if(peak_contract < 0.005){
        cout<<"peak too flat"<<endl;
    }
    else{
       
        // float fine_step = step * 0.5f;//fine_step 0.03
        // cout<<"fine_step"<<fine_step<<endl;
        // z_best = find_bestvalue(z_coarse, fine_step);
    //    if (!csv_header_written) {
    //         csv << "frame,idx,score,alpha,step_norm,z_coarse\n";
    //         // csv_header_written = true;
    //     }
        vector<float> step_range(num_depth_plane);
        for(int i= 0; i < num_depth_plane; i++){
            step_range[i] = (alpha_range[i] - z_coarse) / step;
            // cout<<h_brenner_scores[i]<<","<<alpha_range[i]<<",step,"<< step_range[i]<<endl;
            // csv << frame_id << "," << i << "," <<h_brenner_scores[i]<<","<<alpha_range[i]<<",step,"<< step_range[i]<<"\n";

        }
        // csv<<"\n";
        // frame_id++;
        
        float step_best  = quadratic_fit_points(h_brenner_scores, step_range);
        // float step_best = get_local_5points(h_brenner_scores, max_idx);
        z_best = z_coarse + step_best * step;
        

    }
   
    // cout <<"alpha: "<< z_best << endl;
    da.outputLight(z_best, 2);//turn on the light and measure the latency time

    stage_move(z_best);
    generate_best_view(z_best, max_score);

    // ------------
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("i: %f ms\n", ms);
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    
    
}
void GPUProcessor::stage_move(const float z_best){
    float displacement = 790.13 * (1 - z_best)/400 *100;
    float ad_voltage;
    cout<<"displacement"<<displacement<<endl;
    
    if(fabs(displacement) > THRESHOLD_MIN && fabs(displacement) < THRESHOLD_MAX){
    //MOVE
        if (!ad.IsReady()){
            cout<<"11";
            return;
        }
        if (ad.GetData(ad_voltage) != 0){
            cout<<"222";
            return;
        }
        float current_position = ad_voltage * 10.0f;
        cout<<"current_position"<<current_position<<endl;
        float target_position = current_position - displacement;//-
        cout<<"target position"<<target_position<<endl;
        if (target_position < RANGE_MIN || target_position > RANGE_MAX){
            cout<<"target position is out of the range"<<endl;
            return;
        }
        // float outvoltage = (54 - 50.0f) * 2.0f /10.0f;
        float outvoltage = (target_position - 50.0f) * 2.0f /10.0f;
        cout<<"outvoltage"<<outvoltage<<endl;
        
        da.outputVoltage(outvoltage, 1);

       
        
            
    }else{
        cout<<"displacement is out of the range"<<endl;
        return;

    }
}
float GPUProcessor::quadratic_fit_points(const std::vector<float>& scores, const std::vector<float>& depths) {
    int n = scores.size();
    if (depths.size() != n || n < 3) {
        std::cerr << "Expect at least 3 points!" << std::endl;
        return depths[n/2];
    }
    float max_score = *std::max_element(scores.begin(), scores.end());
    float min_score = *std::min_element(scores.begin(), scores.end());
    float range = max_score - min_score;

    if(range < 1e-6 * max_score){
        cout<<"data is too similar"<<endl;
        auto max_it = std::max_element(scores.begin(), scores.end());
        return depths[std::distance(scores.begin(), max_it)];
    }
    // 构造正规方程的矩阵
    double sum_w = 0, sum_wx = 0, sum_wx2 = 0;
    double sum_wy = 0, sum_wxy = 0, sum_wx2y = 0;
    double sum_wx3 = 0, sum_wx4 = 0;

    for (int i = 0; i < n; ++i) {
        double normalized = (scores[i] - min_score) / range;
        double weight = 0.02 + pow(normalized, 2.0);
        // cout<<"weight:"<<weight<<endl;
        double x = static_cast<double>(depths[i]);
        double y = static_cast<double>(scores[i]);
        
        sum_w += weight;
        sum_wx += weight * x;
        sum_wx2 += weight * x * x;
        sum_wx3 += weight * x * x * x;
        sum_wx4 += weight * x * x * x * x;
        sum_wy += weight * y;
        sum_wxy += weight * x * y;
        sum_wx2y += weight * x * x * y;
    }

    // 3x3 矩阵正规方程: 
    // | Sx4 Sx3 Sx2 |   |A|   = |Sx2y|
    // | Sx3 Sx2 Sx1 | * |B| = |Sxy |
    // | Sx2 Sx1 Sx0 |   |C|   |Sy   |

    // 用克拉默法则求解
    double D = sum_wx4 * (sum_wx2 * sum_w - sum_wx * sum_wx) 
             - sum_wx3 * (sum_wx3 * sum_w - sum_wx * sum_wx2) 
             + sum_wx2 * (sum_wx3 * sum_wx - sum_wx2 * sum_wx2);
    if (fabs(D) < 1e-10) {
        std::cerr << "Singular matrix (D=" << D << "), using max point" << std::endl;
        auto max_it = std::max_element(scores.begin(), scores.end());
        return depths[std::distance(scores.begin(), max_it)];
    }
    // cout<<"D:"<<D<<endl;
    double Da = sum_wx2y * (sum_wx2 * sum_w - sum_wx * sum_wx) 
              - sum_wxy * (sum_wx3 * sum_w - sum_wx * sum_wx2) 
              + sum_wy * (sum_wx3 * sum_wx - sum_wx2 * sum_wx2);
              
    double Db = sum_wx4 * (sum_wxy * sum_w - sum_wy * sum_wx) 
              - sum_wx2y * (sum_wx3 * sum_w - sum_wx * sum_wx2) 
              + sum_wx2 * (sum_wx3 * sum_wy - sum_wx2 * sum_wxy);
    
    // double Dc = sum_wx4 * (sum_wx2 * sum_wy - sum_wx * sum_wx2y)
    //       - sum_wx3 * (sum_wx3 * sum_wy - sum_wx * sum_wx2y)
    //       + sum_wx2 * (sum_wx3 * sum_wxy - sum_wx2 * sum_wx2y);

    

    double A = Da / D;
    double B = Db / D;
    double C = (sum_wy - A * sum_wx2 - B * sum_wx) / sum_w;
    if(A >= 0){
        cout<<"A is larger than 0"<<endl;
        auto max_it = std::max_element(scores.begin(), scores.end());
        return depths[std::distance(scores.begin(), max_it)];
    }
    float best = -B / (2*A);
    double y_best = A * best * best + B * best + C;
    float min_depth = *std::min_element(depths.begin(), depths.end());
    float max_depth = *std::max_element(depths.begin(), depths.end());
    if (best < min_depth - 0.1 || best > max_depth + 0.1) {
        cout<<"the result is out of sampling range"<<endl;
        auto max_it = std::max_element(scores.begin(), scores.end());
        return depths[std::distance(scores.begin(), max_it)];
    }
    // cout<<"best"<<best<<"y_best"<<y_best<<endl;
    return best;
}

float parabolic_3point(const std::vector<float>& scores, int idx){
    int n = scores.size();

    // 必须保证左右都有点
    if (idx <= 0 || idx >= n - 1)
        return 0.0f;

    float yL = scores[idx - 1];
    float y0 = scores[idx];
    float yR = scores[idx + 1];

    float denom = (yL - 2.0f * y0 + yR);

    // 峰太平 or 噪声
    if (std::fabs(denom) < 1e-6f)
        return 0.0f;

    float dx = 0.5f * (yL - yR) / denom;

    // 防止跳太远（经验限制）
    if (dx < -1.0f || dx > 1.0f)
        dx = 0.0f;

    return dx;
}

// float GPUProcessor::find_bestvalue(float z_coarse, const float& fine_step){
//     //second time fitting curve and generate the best depth image
//     // kernel 配置
   
    

//     dim3 block(16, 16);
//     dim3 grid((n_cols+15)/16, (n_rows+15)/16, num_plane);
   

//     vector<float> depth_zn;
//     // depth_zn.assign({z_coarse - 3*fine_step, z_coarse - 2*fine_step,z_coarse - fine_step, z_coarse, z_coarse + fine_step, z_coarse + 2 * fine_step, z_coarse + 3 * fine_step});
//     depth_zn.assign({z_coarse - 2 * fine_step,z_coarse - fine_step, z_coarse, z_coarse + fine_step, z_coarse + 2 * fine_step});
//     cudaMemcpyAsync(d_alpha_n, depth_zn.data(),
//                 sizeof(float) * num_plane,
//                 cudaMemcpyHostToDevice, m_stream);
    

//     shift_and_sum_kernel<<<grid, block, 0, m_stream>>>(d_images, d_volume_n,
//     n_rows, n_cols, patch_size, 
//     d_alpha_n, num_plane);
//     cudaError_t err_shift = cudaGetLastError();
//     if(err_shift != cudaSuccess) printf("shift_Kernel error: %s\n", cudaGetErrorString(err_shift));
//     cudaDeviceSynchronize();
    
    
//     int m = 2;
//     cudaMemsetAsync(d_brenner_scores_n, 0, sizeof(float) * num_plane, m_stream);
//     brenner_kernel<<<grid, block, 0, m_stream>>>(d_volume_n, d_brenner_scores_n, m, n_rows, n_cols, num_plane);
//     cudaError_t err_brenner = cudaGetLastError();
//     if(err_brenner != cudaSuccess) printf("brenner_Kernel error: %s\n", cudaGetErrorString(err_brenner));
//     cudaDeviceSynchronize();
    
//     std::vector<float> h_brenner_scores_n(num_plane);
//     cudaMemcpyAsync(h_brenner_scores_n.data(), d_brenner_scores_n,
//                 sizeof(float) * num_plane,
//                 cudaMemcpyDeviceToHost, m_stream);
//     cudaDeviceSynchronize(); 
    
    
//     vector<float> step_range(num_plane);
//     for(int i= 0; i < num_plane; i++){
//         step_range[i] = (depth_zn[i] - z_coarse) / fine_step;
//         cout<<h_brenner_scores_n[i]<<","<<depth_zn[i]<<",step,"<< step_range[i]<<endl;
//     }

    
//     float step_best  = quadratic_fit_points(h_brenner_scores_n, step_range);
//     float z_refined = z_coarse + step_best * fine_step;
//     cout <<"refined alpha"<< z_refined << endl;
//     return z_refined;
    
 
    
// }



void GPUProcessor::generate_best_view(const float& z_best, float& score){
    dim3 block(16, 16);
    dim3 grid_1((n_cols+15)/16, (n_rows+15)/16, 1);
    cudaMemcpyAsync(d_alpha_1, &z_best,
            sizeof(float) ,
            cudaMemcpyHostToDevice, m_stream);
    shift_and_sum_kernel<<<grid_1, block, 0, m_stream>>>(d_images, d_volume_1,
    n_rows, n_cols, patch_size, 
    d_alpha_1, 1);
    cudaError_t err_shift_1 = cudaGetLastError();
    if(err_shift_1 != cudaSuccess) printf("shift_Kernel error: %s\n", cudaGetErrorString(err_shift_1));
    

    std::vector<float> h_volume(n_rows * n_cols);
    cudaMemcpyAsync(h_volume.data(), d_volume_1, h_volume.size() * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaDeviceSynchronize();
   

    //get 2d coordinate
    std::vector<float> center_img_host(n_rows * n_cols);  
    size_t center_view_idx = static_cast<size_t>(mid_idx) * patch_size + mid_idx;
    size_t offset = center_view_idx * static_cast<size_t>(n_rows) * n_cols;
    // int offset = (mid_idx * patch_size + mid_idx) * n_rows * n_cols;
    cudaMemcpyAsync(center_img_host.data(), d_images + offset, sizeof(float) * n_rows * n_cols, cudaMemcpyDeviceToHost, m_stream);
    cudaError_t err_center = cudaStreamSynchronize(m_stream);
    if(err_center != cudaSuccess) printf("center image copy error: %s\n", cudaGetErrorString(err_center));
    // cudaDeviceSynchronize();
    // cv::Mat center_img(n_rows, n_cols, CV_32FC1, center_img_host.data());
    // cv::Mat center_show;
    // center_show.convertTo(center_show, CV_8UC1, 255);

    //save the data into the buffer
    auto* wBuf = resultBuffer.writeBuf.load();
    wBuf->best_image = h_volume;
    wBuf->best_value = z_best;
    wBuf->center_img = center_img_host;

    auto* oldRead = resultBuffer.readBuf.load();
    resultBuffer.readBuf.store(wBuf);
    resultBuffer.writeBuf.store(oldRead);

    resultBuffer.newResult = true;
    resultBuffer.cv.notify_one();

    
}


// vector<cv::Vec3f> GPUProcessor::currentimage(){
//     int threads = 256;
//     int total_threads = n_rows * n_cols ;
//     int transform_blocks = (total_threads + threads - 1) / threads;
//     transform_kernel_centerview<<<transform_blocks, threads, 0, m_stream>>>(
//         d_imagefloat, image_rows, image_cols, 
//         d_rows_flat, d_rows_offsets,
//         total_circles, n_rows, n_cols, patch_size,
//         d_image_1
//     );
//     cudaError_t err_transform_center = cudaGetLastError();
//     if(err_transform_center != cudaSuccess) printf("transform_Kernel error: %s\n", cudaGetErrorString(err_transform_center));
//     cudaDeviceSynchronize();
//     std::vector<cv::Vec3f> h_image( n_rows * n_cols);
  
//     cudaMemcpyAsync(h_image.data(), d_image_1,
//     h_image.size() * sizeof(cv::Vec3f),
//     cudaMemcpyDeviceToHost, m_stream);
//     cudaDeviceSynchronize();
    
//     return h_image;
// }

// void GPUProcessor::test(){
//     dim3 block(16, 16);
//     dim3 grid((n_cols+15)/16, (n_rows+15)/16, num_depth_plane);
//     cv::Mat black_image(image_rows, image_cols, CV_8UC3, cv::Scalar(0, 0, 0));
//     cuda_preprocess(black_image);
//     int m = 2;
//     cudaMemset(d_brenner_scores, 0, num_depth_plane * sizeof(float));
//     brenner_kernel<<<grid, block, 0, m_stream>>>(d_imagefloat, d_brenner_scores, m, n_rows, n_cols, num_depth_plane);
//     cudaError_t err_brenner = cudaGetLastError();
//     if(err_brenner != cudaSuccess) printf("brenner_Kernel error: %s\n", cudaGetErrorString(err_brenner));
//     cudaDeviceSynchronize();
    
//     std::vector<float> h_brenner_scores2(num_depth_plane);
//     cudaMemcpyAsync(h_brenner_scores2.data(), d_brenner_scores,
//                 sizeof(float) * num_depth_plane,
//                 cudaMemcpyDeviceToHost, m_stream);
//     for(int i = 0; i < num_depth_plane; i++) {
//         cout << "Depth " << i << ": " << h_brenner_scores2[i] << endl;
//     }
//     cv::Mat checkborad(image_rows, image_cols, CV_8UC3);
//     for(int i = 0; i < image_rows; i++) {
//         for(int j = 0; j < image_cols; j++) {
//             cv::Scalar color = ((i/10 + j/10) % 2 == 0) ? cv::Scalar(255,255,255) : cv::Scalar(0,0,0);
//             checkborad.at<cv::Vec3b>(i,j) = cv::Vec3b(color[0], color[1], color[2]);
//         }
//     }
//     cuda_preprocess(checkborad);
//     cudaMemset(d_brenner_scores, 0, num_depth_plane * sizeof(float));
//     brenner_kernel<<<grid, block, 0, m_stream>>>(d_imagefloat, d_brenner_scores, m, n_rows, n_cols, num_depth_plane);
   
   
    
//     std::vector<float> h_brenner_scores3(num_depth_plane);
//     cudaMemcpyAsync(h_brenner_scores3.data(), d_brenner_scores,
//                 sizeof(float) * num_depth_plane,
//                 cudaMemcpyDeviceToHost, m_stream);
//     for(int i = 0; i < num_depth_plane; i++) {
//         cout << "Depth2 " << i << ": " << h_brenner_scores3[i] << endl;
//     }
//     cudaMemset(d_brenner_scores, 0, num_depth_plane * sizeof(float));
    
//     cv::Mat gradient(image_rows, image_cols, CV_8UC3);
//     for(int i = 0; i < image_rows; i++) {
//         for(int j = 0; j < image_cols; j++) {
//             int val = (j * 255) / image_cols;
//             gradient.at<cv::Vec3b>(i,j) = cv::Vec3b(val, val, val);
//         }
//     }
//     cuda_preprocess(gradient);
    
//     brenner_kernel<<<grid, block, 0, m_stream>>>(d_imagefloat, d_brenner_scores, m, n_rows, n_cols, num_depth_plane);
//     cudaStreamSynchronize(m_stream);
    
//     std::vector<float> scores_gradient(num_depth_plane);
//     cudaMemcpy(scores_gradient.data(), d_brenner_scores, sizeof(float) * num_depth_plane, cudaMemcpyDeviceToHost);
    
//     for(int i = 0; i < num_depth_plane; i++) {
//         cout << "Depth3 " << i << ": " << scores_gradient[i] << " (should be moderate)" << endl;
//     }
// }
void GPUProcessor::prepare_data(){
    rows_flat.clear();
    rows_offsets.clear();
    //flatten the uploaded data
    int offsets = 0;
    for(auto& row : rows){
        rows_offsets.push_back(offsets);
        rows_flat.insert(rows_flat.end(), row.begin(), row.end());
        offsets += row.size();
    }
    total_circles = static_cast<int>(rows_flat.size());
    // cout<<"rows_flat"<<rows_flat.size()<<endl;
}
void GPUProcessor::extract_rows(const vector<CircleInf>& circleList, const float y_tolerance){
    vector<CircleInf> sortedList;
    // preprocess(circles, sortedList);
    //sort by the y-axis
    sortedList = circleList;
    sort(sortedList.begin(), sortedList.end(),
    [](const CircleInf& a, const CircleInf& b) { return a.y < b.y; });
    int estimated_rows = max(1, (int)(sortedList.size()/30));
    rows.clear();
    rows.reserve(estimated_rows);
    while (!sortedList.empty())
    {
        /* code */
        float row_y = sortedList[0].y;
        for (const auto& c : sortedList)
        {
            /* code */
            if (c.y < row_y)
                row_y = c.y;
        }
        vector<CircleInf>current_row;
        // current_row.reserve(34);
        vector<CircleInf>remaining_row;
        remaining_row.reserve(sortedList.size());
        for (const auto& c : sortedList) {
            if (abs(row_y - c.y) < y_tolerance)
                current_row.push_back(c);
            else remaining_row.push_back(c);
        }
        //sort by x-axis
        sort(current_row.begin(), current_row.end(), [](const CircleInf& a, const CircleInf& b) {
            return a.x < b.x;
            });
        rows.push_back(current_row);
        // cout<< current_row.size()<<endl;
        sortedList = move(remaining_row);

    }
    size_t max_cols = 0;
    for (const auto& row : rows)
    {
        max_cols = max(max_cols, row.size());
    }
    for(auto& row : rows){
        if(row.size() < max_cols){
            row.resize(max_cols);
        }
    }
    n_cols = static_cast<int>(max_cols);
    n_rows =  static_cast<int>(rows.size());
    // cout<<"rows size"<<rows.size()<<endl;
    // cout<<"cols size"<<n_cols<<endl;
    
}
void GPUProcessor::cuda_preprocess(const cv::Mat& image_mla){
 
    CV_Assert(image_mla.isContinuous());
    d_img.upload(image_mla);
    cudaError_t err_upload = cudaGetLastError();
    if(err_upload != cudaSuccess) printf("copy_Kernel error: %s\n", cudaGetErrorString(err_upload));
    uchar* d_img_ptr = d_img.ptr<uchar>();
   
    
    
    dim3 block_copy(16, 16);
    dim3 grid_copy((image_cols + 15)/16, (image_rows + 15)/16);
    // 分配输出 device 内存（用 float*，回传时直接拷回到 Vec3f 数组）
    
    size_t step = d_img.step;
    copy_kernel<<<grid_copy, block_copy, 0, m_stream>>>(d_img_ptr, image_rows, image_cols, step, d_imagefloat);
    cudaDeviceSynchronize();
    cudaError_t err_copy = cudaGetLastError();
    if(err_copy != cudaSuccess) printf("copy_Kernel error: %s\n", cudaGetErrorString(err_copy));
  
}

// // void GPUProcessor::preprocess(const vector<cv::Vec3f>& circles, vector<CircleInf>& sortedList){
//     vector<CircleInf> circleList;
//     //orgnaize the array
//     for (int i = 0; i < circles.size(); i++)
//     {
//         float x = std::round(circles[i][0] * 100) / 100.0f;
//         float y = std::round(circles[i][1] * 100) / 100.0f;
//         float radius = std::round(circles[i][2] * 100) / 100.0f;
//         circleList.push_back({ x, y, radius });
        
//     }
//     //sort by the y-axis
  
//     vector<int> idx(circleList.size());
//     for (int i = 0; i < idx.size(); i++) idx[i] = i;
//     sort(idx.begin(), idx.end(), [&](int a, int b) {
//         return circleList[a].y < circleList[b].y;
//     });
    
//     for (int i = 0; i < idx.size(); i++)
//     {
//         /* code */
//         sortedList.push_back(circleList[idx[i]]);
//     }
    
//     //set the range
//     int rangex1 = 900, rangex2 = 3000;
//     int rangey1 = 125, rangey2 = 2000;
    
//     vector<CircleInf> rangeList;
//     for (int i = 0; i < sortedList.size(); i++)
//     {
//         /* code */
//         float x = sortedList[i].x;
//         float y = sortedList[i].y;
        
//         if (x >= rangex1 && x <= rangex2 && y >= rangey1 && y <= rangey2) {
//             rangeList.push_back(sortedList[i]);
//         }
//     }
//     sortedList = move(rangeList);

// }