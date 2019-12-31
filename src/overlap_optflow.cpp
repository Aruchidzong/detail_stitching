//
// Created by aruchid on 2019/12/25.
//



#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>



#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

//#include "../stitch_connector.h"
#include "../src_out/optical.hpp"
//#include "../include/aruchid_featuresfind.hpp"
// #include "extra.h" // use this if in OpenCV2
using namespace std;
using namespace cv;
using namespace cv::detail;

bool start_flag = true;
bool show_flag = true;

cv::Mat color, last_color;
std::vector<Point> scene_corners_(4);
////for 找回


int main( int argc, char** argv )
{
    cv::VideoCapture cap;
//    cap.open("1.mp4");
//    cap.open("2.mp4");
//    cap.open("3.mp4");
//    cap.open("4.mp4");
//    cap.open("5.mp4");
//    cap.open("6.mp4");
//    cap.open("7.mp4");
//    cap.open( "s1.mp4");
//    cap.open("s2.mp4");
//    cap.open("9.mp4");
//    cap.open("10.mp4");
//    cap.open("11.mp4");
//    cap.open("27/4.mp4");
    cap.open(0);
    // 判断摄像头是否打开
    if (!cap.isOpened()){
        std::cerr << "Could't open capture" << std::endl;
        return -1;
    }
    vector<Mat> pic_;
    vector<Mat> homos_;
    int Max_origin_size;
    char keyCode;
    Scalar showScalar = Scalar(0,255,0);
    int frame_nums=0;
    for(;;)
    {
        cap >> color;
        Mat temp;
        color.copyTo(temp);
        if(color.empty())
            break;

        keyCode = cv::waitKey(20);
        if (keyCode == 's') {
            start_flag = true;
        }
        if (start_flag)
        {
            //清楚所有flag
            set_src_feature(temp);
            start_flag = false;
            continue;
        }
        if ( color.data==nullptr)
            continue;
        if(!start_flag){
            show_flag = overlap_point(temp,scene_corners_,scene_corners_);
        }
        cout << endl;
        Mat dst_, src_;
        color.copyTo(src_);
        src_.copyTo(dst_);

        if(show_flag){
            cv::Point pt[1][4];
            pt[0][0] = scene_corners_[3];
            pt[0][1] = scene_corners_[0];
            pt[0][2] = scene_corners_[1];
            pt[0][3] = scene_corners_[2];
            const cv::Point *ppt[1] = {pt[0]};
            int npt[1] = {4};
            line(dst_, scene_corners_[0], scene_corners_[1], Scalar(0, 240, 0), 5);
            line(dst_, scene_corners_[1], scene_corners_[2], Scalar(0, 240, 0), 5);
            line(dst_, scene_corners_[2], scene_corners_[3], Scalar(0, 240, 0), 5);
            line(dst_, scene_corners_[3], scene_corners_[0], Scalar(0, 240, 0), 5);

        }

        cv::namedWindow("addweight", cv::WINDOW_NORMAL);
        cv::imshow("addweight", dst_);
        cv::waitKey(1);
    }



    return 0;
}