#include "refocus_cmake.h"





int main()
{
    cout << "filepath: " << std::filesystem::current_path().string() << std::endl;
    Mat image = imread("data/Image__2025-05-23__16-37-19.bmp");
    Mat image_mla = imread("data/original_20250617_180038.bmp");
    Mat image_rgb;
    Mat image_gray;
    int Rmin = 9, Rmax = 30;

    if (image.empty()) {
        //throw runtime_error("No image, please check the source");
        cout << "no image, please check." << endl;
        return -1;
    }
    if (image_mla.empty()) {
        throw runtime_error("No image_mla, please check the source");
        cout << "no image, please check." << endl;
        return -1;
    }
    
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    cvtColor(image_mla, image_rgb, COLOR_BGR2RGB);
    
    //hough transform
    vector<Vec3f> circles;
    HoughCircles(image_gray, circles, HOUGH_GRADIENT, 1.2, 28, 50, 30, Rmin, Rmax);
    vector<CircleInf> circleList;
    
    
    if (!circles.empty()) {
        for (int i = 0; i < circles.size(); i++)
        {
            float x = std::round(circles[i][0] * 100) / 100.0f;
            float y = std::round(circles[i][1] * 100) / 100.0f;
            Point2f center(x, y);  
            float radius = round(circles[i][2] * 100) / 100.0f;
            circleList.push_back({ center, radius });
            
        }
        
        //sort by the y-axis
        vector<int> idx(circleList.size());
        for (int i = 0; i < idx.size(); i++) idx[i] = i;
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return circleList[a].center.y < circleList[b].center.y;
        });
        vector<CircleInf> sortedList;
        for (int i = 0; i < idx.size(); i++)
        {
            /* code */
            sortedList.push_back(circleList[idx[i]]);
        }
        
        //set the range
        int rangex1 = 900, rangex2 = 3000;
        int rangey1 = 125, rangey2 = 2000;
        
        vector<CircleInf> rangeList;
        for (int i = 0; i < sortedList.size(); i++)
        {
            /* code */
            float x = sortedList[i].center.x;
            float y = sortedList[i].center.y;
            
            if (x >= rangex1 && x <= rangex2 && y >= rangey1 && y <= rangey2) {
                rangeList.push_back(sortedList[i]);
            }
        }
        sortedList = move(rangeList);
        
        
        
        //extract the patch image to the array
        int patch_size = 64;
        float tolerance = 15;
        //rows: collect the circle centers and sort by the y axis and x axis
        vector<vector<CircleInf>> rows = extract_rows(sortedList, tolerance);
        // //patches: cut each microlens images based on each circle center and collect them into the two dimension array
        // vector<vector<Mat>>patches = extract_microlens_patches(image_mla, patch_size, rows);
        // vector<vector<vector<vector<Vec3b>>>> vp_img_arr;
        vector<Mat> views;
        int n_rows, n_cols;
        // int patch_h, patch_w;
        //vp_img_arr: change the sequence from [n_rows, n_cols, patch_h,patch_w] to [patch_h,patch_w,n_rows,n_cols]
        //images: the images array of each angle
        // transpose_patch_array(patches, vp_img_arr, images, n_rows, n_cols, patch_h, patch_w);
        //    cout<<"111";
        tranform_image(image_mla, patch_size, rows, views, n_rows, n_cols);
        
        
        
        
        //refocus processing optical parament
        int num_depth_plane = 1;
        float start = -5, end = 0;
        //create the disparity table
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
        vector<vector<vector<float>>> disparity_x(patch_size, vector<vector<float>>(patch_size, vector<float>(num_depth_plane, 0.0f)));
        vector<vector<vector<float>>> disparity_y(patch_size, vector<vector<float>>(patch_size, vector<float>(num_depth_plane, 0.0f)));
        
        for(int z_idx = 0; z_idx < num_depth_plane; z_idx++){
            float z = depth_range[z_idx];
            float factor = s * (z / (f + z)) / pixel_size;
            for(int h = 0; h < patch_size; h++){
                for(int w = 0; w < patch_size; w++){
                    disparity_x[h][w][z_idx] = (w - mid_idx) * factor ;
                    disparity_y[h][w][z_idx] = (h - mid_idx) * factor ;
                }
            }
        }
        //volume: refocused image array
        vector<Mat> volume = shift_and_sum(views, n_rows, n_cols, num_depth_plane, patch_size, disparity_x, disparity_y);
        auto start1 = chrono::high_resolution_clock::now();
        auto end1 = chrono::high_resolution_clock::now(); 
        
        auto duration1 = duration_cast<chrono::milliseconds>(end1 - start1);
        cout << "running time：" << duration1.count() << " milliseconds" << endl;    
        
        // for (size_t i = 0; i < volume.size(); i++)
        // {
            //     /* code */
        //     imshow("image",volume[i]);
        //     waitKey(0);
        // }
        
   
    }
    
    

    return 0;

}

vector<vector<CircleInf>>extract_rows(vector<CircleInf> sortedList, float y_tolerance) {
    
    vector<vector<CircleInf>> rows;
    int estimated_rows = max(1, (int)(sortedList.size()/64));
    rows.reserve(estimated_rows);
    while (!sortedList.empty())
    {
        /* code */
        float row_y = sortedList[0].center.y;
        for (const auto& c : sortedList)
        {
            /* code */
            if (c.center.y < row_y)
                row_y = c.center.y;
        }
        vector<CircleInf>current_row;
        current_row.reserve(64);
        vector<CircleInf>remaining_row;
        remaining_row.reserve(sortedList.size());
        for (const auto& c : sortedList) {
            if (abs(row_y - c.center.y) < y_tolerance)
                current_row.push_back(c);
            else remaining_row.push_back(c);
        }
        //sort by x-axis
        sort(current_row.begin(), current_row.end(), [](const CircleInf& a, const CircleInf& b) {
            return a.center.x < b.center.x;
            });
        rows.push_back(current_row);
        sortedList = move(remaining_row);

    }
     
    return rows;
}

void tranform_image(const Mat& image_mla, const int patch_size, const vector<vector<CircleInf>>& rows, vector<Mat>& views, int &n_rows, int &n_cols){
    auto inner_start = chrono::high_resolution_clock::now();
    assert(patch_size > 0);
    n_rows = static_cast<int>(rows.size());
    n_cols = 0;
    float half = (patch_size - 1) / 2.0f;
    for(const auto& row : rows){
        n_cols = max(n_cols, (int)row.size());
    }
    views.resize(patch_size * patch_size);
    assert(views.size() == patch_size * patch_size);
    for (int i = 0; i < views.size(); i++)
    {
        views[i].create(n_rows, n_cols, CV_32FC3);
    }
    vector<Point2f> offset(patch_size * patch_size);
    
    for (int u = 0; u < n_rows; u++)
    {
        for (int v = 0; v < rows[u].size(); v++)
        {
            Point2f center = rows[u][v].center;
            for (int s = 0; s < patch_size; s++)//patch_h
            {
                
                for(int t = 0; t < patch_size; t++){//patch_w
                    int view_idx = s * patch_size + t;
                    float offset_x = t - half;
                    float offset_y = s - half;
                    Point2f sample_point = center + Point2f(offset_x, offset_y);
                    Vec3f color = bilinear_rgb(image_mla, sample_point);
                    Vec3f* row_ptr = views[view_idx].ptr<Vec3f>(u);
                    row_ptr[v] = color;
                }
            }
            
        }
        
    }
    auto inner_end = chrono::high_resolution_clock::now();
    auto inner_dur = duration_cast<chrono::nanoseconds>(inner_end - inner_start).count();
    cout<<inner_dur<<endl;
   
}
Vec3b bilinear_rgb(const Mat& image_mla, const Point2f& pt){
    int x = static_cast<int>(floor(pt.x));
    int y = static_cast<int>(floor(pt.y));
    float dx = pt.x - x;
    float dy = pt.y - y;
    if(x < 0 || x + 1 >= image_mla.cols || y < 0 || y + 1 >= image_mla.rows)
    return Vec3b(0, 0, 0);
    const Vec3b* row0 = image_mla.ptr<Vec3b>(y);
    const Vec3b* row1 = image_mla.ptr<Vec3b>(y + 1);
    
    Vec3b I00(row0[x]);
    Vec3b I01(row0[x + 1]);
    Vec3b I10(row1[x]);
    Vec3b I11(row1[x + 1]);
    
    
    return (1 - dx) * (1 - dy) * I00 +
    dx * (1 - dy) * I01 +
    (1 - dx) * dy * I10 +
    dx * dy * I11;
}
vector<Mat> shift_and_sum(const vector<Mat> &images, const int& n_rows, const int& n_cols, const int& num_depth_plane,const int& patch_size, const vector<vector<vector<float>>>& disparity_x, const vector<vector<vector<float>>>& disparity_y){

    
    vector<Mat> volume(num_depth_plane);
    Mat shifted(images[0].size(), CV_32FC3);
    cout << "image type: " << images[0].type() << endl;
    for (int z_idx = 0; z_idx < num_depth_plane; z_idx++)
    {   
        /* code */
        Mat refcoused(n_rows, n_cols, CV_32FC3, Scalar(0,0,0));
        for (int h = 0; h < patch_size; h++)
        {
            /* code */
            for (int w = 0; w < patch_size; w++)
            {
                /* code */
                // Mat img_float;
                // images[h][w].convertTo(img_float, CV_32FC3);
                float dx = disparity_x[h][w][z_idx];
                float dy = disparity_y[h][w][z_idx];
                
                shift_img(images[h * patch_size + w], dx, dy, shifted);
                
                refcoused += shifted;
                
            }
            
        }
        refcoused /= static_cast<float>(patch_size * patch_size);
        Mat refocused_unit8;
        refcoused.convertTo(refocused_unit8, CV_8UC3, 1.0, 0.0);
        
        float scale = 10.0f;
        Mat large;
        cv::resize(refocused_unit8, large, Size(), scale, scale, INTER_LINEAR);
        volume[z_idx] = large;
    }
   
    return volume;
}
void shift_img(const Mat& img, float dx, float dy, Mat& shifted){
    
  
    //Affine transformation matrix
    Mat M = (Mat_<double>(2,3)<<1, 0, dx, 0 , 1, dy);
    warpAffine(img, shifted, M, img.size(), INTER_LINEAR, BORDER_REFLECT);
  
}
// vector<vector<Mat>>extract_microlens_patches(Mat image_mla, int patch_size, vector<vector<CircleInf>>rows){
    //     size_t n_rows = rows.size();
    //     int n_cols = 0;
//     for(const auto& row : rows){
//         n_cols = max(n_cols, (int)row.size());
//     }
//     vector<vector<Mat>> patches(n_rows,vector<Mat>(n_cols));//定义patches数组
    
    
//     for (int i = 0; i < n_rows; i++)
//     {
//         /* code */
//         for (int j = 0; j < (int)rows[i].size(); j++)
//         {
//             /* code */
//             patches[i][j] = extract_patch(image_mla, rows[i][j].center, patch_size) ;
//         }

//     }
//     return patches;
// }




// Mat extract_patch(const Mat& image, const Point2f& center, int patch_size){
//     static int half = patch_size / 2;
//     static Mat patch;
//     static Mat map_x(patch_size, patch_size, CV_32FC1);
//     static Mat map_y(patch_size, patch_size, CV_32FC1);

//     for (int y = 0; y < patch_size; y++)
//     {
//         /* code */
//         for (int x = 0; x < patch_size; x++)
//         {
//             /* code */
//             static float xf = center.x - half + 0.5f + x;
//             static float yf = center.y - half + 0.5f + y;
//             map_x.at<float>(y, x) = xf;
//             map_y.at<float>(y, x) = yf;
//         }

//     }
//     remap(image, patch, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
//     return patch;
// }

// void transpose_patch_array(const vector<vector<Mat>>& patches, vector<vector<vector<vector<Vec3b>>>> &vp_img_arr, vector<vector<Mat>> &images, int &n_rows, int &n_cols, int &patch_h, int &patch_w){
//     n_rows = static_cast<int>(patches.size());
//     n_cols = 0;
//     for(const auto& row : patches){
//         n_cols = max(n_cols, (int)row.size());
//     }
//     if (n_rows == 0 || n_cols == 0 || patches[0][0].empty()) {
//         cout << "Invalid patch array!" << endl;
//         return;
//     }
//     patch_h = patches[0][0].rows;
//     patch_w = patches[0][0].cols;

//     //set the zeros array vp_img_arr
//     vp_img_arr = vector<vector<vector<vector<Vec3b>>>>(patch_h, 
//         vector<vector<vector<Vec3b>>>(patch_w, vector<vector<Vec3b>>
//             (n_rows, vector<Vec3b>(n_cols))));
//     for(int u = 0; u < n_rows; u++){
//         for(int v = 0; v < patches[u].size(); v++){
//             const Mat& patch = patches[u][v];
//             // cout<<patches[i].size();
//             // cout<<"i:"<< i<<"j:"<<j<<endl;
//             if(patch.empty() || patch.rows != patch_h || patch.cols != patch_w ||
//             patch.channels() != 3){
//                 cout<<"skip invaild patch at ["<<u<<"]["<<v<<"]"<<endl;
//                 continue;
//             }
            
//             for(int s = 0 ; s < patch_h; s++){
//                 for(int t = 0; t < patch_w; t++){
                 
//                     vp_img_arr[s][t][u][v] = patch.at<Vec3b>(s,t);
//                 }
//             }
         
//         }
//     }

//     //convert vp_img_arr to whole mat corresponding on each [u][v]
//     images = vector<vector<Mat>>(patch_h, vector<Mat>(patch_w));
//     for(int s = 0; s < patch_h; s++){
//         for (int t = 0; t < patch_w; t++)
//         {
//             /* code */
//             //initialization
//             // Mat img(n_rows, n_cols, CV_8UC3, Scalar(0,0,0));
//             Mat img(n_rows, n_cols, CV_32FC3, Scalar(0,0,0));
//             for (int u = 0; u < n_rows; u++)
//             {
//                 /* code */
//                 for(int v = 0; v < n_cols; v++){
//                     if(v < patches[u].size() && !patches[u][v].empty() && patches[u][v].channels() == 3){
//                         // img.at<Vec3b>(u,v) = vp_img_arr[s][t][u][v];
//                         img.at<Vec3f>(u, v) = Vec3f(
//                         vp_img_arr[s][t][u][v][0],
//                         vp_img_arr[s][t][u][v][1],
//                         vp_img_arr[s][t][u][v][2]
//                         );
//                     }

//                 }
//             }
//             images[s][t] = img;
//         }
         
//     }

// }

// void shift_img_implace(const Mat& img, float dx, float dy, Mat& out){
//     // out.setTo(Scalar(0,0,0));
//     int n_rows = img.rows;
//     int n_cols = img.cols;
//     for(int i = 0; i < n_rows; i++){
//         for(int j = 0; j < n_cols; j++){
//             float x = j + dx;
//             float y = i + dy;
//             Vec3f color = bilinear_rgb(img, Point2f(x, y));
//             Vec3f* row_ptr = out.ptr<Vec3f>(i);
//             row_ptr[j] = color;
//         }
//     }
// }

