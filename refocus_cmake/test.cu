#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>


struct CircleInf { float x, y, r; int valid; };


int main() {
    // std::vector<CircleInf> v = {{1,2,3,1}, {4,5,6,1}};
    // for (auto &c : v) std::cout << c.x << "\n";
    int n = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << n << std::endl;
    return 0;
}
