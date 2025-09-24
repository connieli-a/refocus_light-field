#include <algorithm>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h> 
#include <chrono>

using namespace Pylon;
using namespace GenApi; // NodeMap / CFloatPtr / CEnumerationPtr


int main() {
     // The exit code of the sample application
    int exitCode = 0;
    int screenWidth = 1536; // 屏幕宽
    int screenHeight = 864; // 屏幕高
    bool flag = false;
    int rangex1 = 900, rangex2 = 3000;//960 2880
    int rangey1 = 200, rangey2 = 1800;//540 1620
    static bool firstFrame = true;
    // Before using any pylon methods, the pylon runtime must be initialized.
    PylonInitialize();
    try{
        //open camera
        CInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice());
        camera.Open();
        
        INodeMap& nodemap = camera.GetNodeMap();
        //exposure time
        CFloatPtr exposureTime(nodemap.GetNode("ExposureTime"));
        if(IsWritable(exposureTime))
            exposureTime->SetValue(200.0);//µs
        CIntegerPtr width(nodemap.GetNode("Width"));
        CIntegerPtr height(nodemap.GetNode("Height"));
        CIntegerPtr offsetx(nodemap.GetNode("OffsetX"));
        CIntegerPtr offsety(nodemap.GetNode("OffsetY"));
        CEnumerationPtr balancesector(nodemap.GetNode("BalanceWhiteAuto"));
        if(IsWritable(width)) width->SetValue(rangex2 - rangex1);
        if(IsWritable(height)) height->SetValue(rangey2 - rangey1);
        if(IsWritable(offsetx)) offsetx->SetValue(rangex1);
        if(IsWritable(offsety)) offsety->SetValue(rangey1);
        // if(IsWritable(balancesector)) balancesector->FromString("Off");
        

        camera.StartGrabbing(GrabStrategy_OneByOne);
        CGrabResultPtr ptrGrabResult;
        static std::chrono::high_resolution_clock::time_point last;
        while(camera.IsGrabbing()){
            camera.RetrieveResult(INFINITE, ptrGrabResult, TimeoutHandling_ThrowException);
            
            auto end = std::chrono::high_resolution_clock::now();
            if(ptrGrabResult->GrabSucceeded()){
                CPylonImage pylonImage;
                CImageFormatConverter converter;
                converter.OutputPixelFormat = PixelType_RGB8packed;
                converter.Convert(pylonImage, ptrGrabResult);
                
                auto start = std::chrono::high_resolution_clock::now();
                if(firstFrame){
                    last = start;
                    firstFrame = false;
                }
                auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(start - last).count();
                last = start;
                cv::Mat frame(static_cast<int>(ptrGrabResult->GetHeight()), static_cast<int>(ptrGrabResult->GetWidth()), CV_8UC3, (uint8_t*)pylonImage.GetBuffer());
                // double scaleX = screenWidth / double(frame.cols);
                // double scaleY = screenHeight / double(frame.rows);
                // std::cout<<"cols:"<<frame.cols<<"rows"<<frame.rows<<std::endl;
                // double scale = min(scaleX, scaleY); // 保证完整显示
                // cv::Mat display;
               
                // cv::resize(frame, display, cv::Size(), scale, scale); // 使用统一缩放比例
                // cv::namedWindow("Camera", cv::WINDOW_NORMAL);
                // if(!flag){
                //     cv::imwrite("image.bmp",frame);
                //     std::cout<<"saved one image"<<std::endl;
                //     flag = true;
                // }
                cv::Mat frameBGR;
                cv::cvtColor(frame, frameBGR, cv::COLOR_RGB2BGR);
                cv::imshow("Camera", frameBGR);
                if (cv::waitKey(1) == 27) // ESC 退出
                    break;

                // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout<<"grab one image:"<<frame_time<<std::endl;
            }
        }
        camera.StopGrabbing();
        camera.Close();
    }
    catch(const GenericException &e){
        std::cerr<<"An exception occurued:"<<e.GetDescription()<<std::endl;
    }
    PylonTerminate();
    return 0;
}
