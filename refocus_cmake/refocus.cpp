#include "include/refocus.h"
#include "include/cuda_kernel.cuh"

#include <iostream>

//  * @param device 使用するGPUのID (デフォルトは0)
//  * @param useGraph CUDA Graphを使用するかどうか (デフォルトはtrue)
//  * @return ImageProcessorのインスタンス
//  default mode(use gpu)->useGraph: true
std::shared_ptr<ImageProcessor> ImageProcessor::create(const vector<cv::Vec3f> circles, const float y_tolerance, const int patch_size_cpp, const int num_depth_plane, const std::vector<float> disparity_x_flat, const std::vector<float> disparity_y_flat, const int32_t device, const bool useGraph){
     //only 1 gpu device
    // int device_count = 0;
//     cudaGetDeviceCount(&device_count);
//     std::cout << "Detected " << device_count << " CUDA devices." << std::endl;
//     if (device < 0 || device >= device_count) {
//     std::cerr << "Error: invalid GPU device id " << device 
//               << ", available range is 0 to " << device_count - 1 << std::endl;
//     exit(EXIT_FAILURE);
// }

    cudaSetDevice(device);
    
    // int current_device;
    // cudaGetDevice(&current_device);
    // std::cout<<"using gpu"<<current_device<<std::endl;
    if(useGraph){
        return std::make_shared<GPUProcessor>(circles, y_tolerance, patch_size_cpp, num_depth_plane, disparity_x_flat, disparity_y_flat);
    }
   return nullptr;
}




  
void generate_disparity_table(const int num_depth_plane, const int start, const int end, const int patch_size, vector<float>& disparity_x_flat, vector<float>& disparity_y_flat){
    int mid_idx = patch_size / 2;
    float pixel_size = 2.0;
    int s = 125;//the diameter of the lens
    int f = 2500;// the focal length of the lens
    vector<float> depth_range;
    
    //generate depth_range
    if(num_depth_plane == 1)
        depth_range.push_back(start);
    else{
        float step = static_cast<float>(end - start) / static_cast<float>(num_depth_plane - 1);
        for (int i = 0; i < num_depth_plane; ++i)
        {
            /* code */
            depth_range.push_back(start + step * i);
        }
    }
    for(int z_idx = 0; z_idx < num_depth_plane; z_idx++){
        float z = depth_range[z_idx];
        float factor = s * (z / (f + z)) / pixel_size;
        for(int h = 0; h < patch_size; h++){
            for(int w = 0; w < patch_size; w++){
                int idx = (h * patch_size + w) * num_depth_plane + z_idx;//disparity_x[h][w][z_idx]
                disparity_x_flat[idx] = (w - mid_idx) * factor ;
                disparity_y_flat[idx] = (h - mid_idx) * factor ;
            }
        }
    }
}
