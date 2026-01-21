#include "include/refocus.h"
#include "include/DAControl.h"
#include "include/ADControl.h"
FrameBuffer frameBuffer;
ResultBuffer resultBuffer;
DisplayBuffer displayBuffer;
CameraDisplayBuffer cameraDisplayBuffer;
std::atomic<bool> running {true};
//set the range
int rangex1 = 180, rangex2 = 540;//960 3000
int rangey1 = 90, rangey2 = 450;//200 1800
cv::Mat image_mla_roi;

DAControl da;
ADControl ad;