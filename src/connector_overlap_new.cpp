//
// Created by aruchid on 2019/12/25.
//

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

//#include "./aruchid_log.h"
// #include "extra.h" // use this if in OpenCV2
using namespace std;
using namespace cv;
using namespace cv::detail;


int Max_origin_Size = 100;
bool start_flag = true;

//vector<KeyPoint> last_photo_points;
//std::vector<Point2f> scene_corners(4);

Mat H_last_ = Mat::eye(3,3,CV_64FC1);
Mat H_optflow_ = Mat::eye(3,3,CV_64FC1);
list<cv::Point2f> origin_keypoints; //只是一个list存储所有的初始点
list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
cv::Mat color, last_color;
//for 找回
//vector<KeyPoint> start_keypoints;
//Mat start_Descriptor;

vector<Point2f>scene_corners(4);
bool Calc_Overlap(Mat frame,vector<Point2f>&cornerOutput);
bool Start_Overlap(Mat template_photo);

Size Get_Overlap_Area(vector<Point2f> corners);
void Trans_H();

int main( int argc, char** argv )
{
    cv::VideoCapture cap;
//    cap.open("1.mp4");
//    cap.open("2.mp4");
//    cap.open("3.mp4");
//    cap.open("4.mp4");
    cap.open("5.mp4");
//    cap.open("6.mp4");
//    cap.open("7.mp4");
//    cap.open("VID_20191223_172716.mp4");

//    cap.open(0);
    // 判断摄像头是否打开
    if (!cap.isOpened()){
        std::cerr << "Could't open capture" << std::endl;
        return -1;
    }
    char keyCode;

    for(;;)
    {
        cap >> color;
        if(color.empty())
            break;


        resize(color ,color ,Size() ,0.5,0.5);
        rotate(color,color,ROTATE_90_CLOCKWISE);
        cout<< color.size << endl;

        keyCode= cv::waitKey(30);
        //todo 这里是拍照调用的接口
        if (keyCode == 's')
            start_flag = true;
        if (start_flag)
        {
            //清chu所有flag
            Start_Overlap(color);
            continue;
        }
        if ( color.data==nullptr)
            continue;

//        if(lost_flag)
        Calc_Overlap(color,scene_corners);
//        if(!lost_flag)
//            lost_flag = Retrieve_Overlap();

        Scalar showScalar = Scalar(0,255,0);
//        for(auto corner:scene_corners){
//            if((corner.x<-color.rows*2/2)||(corner.x>color.rows*2))
//                showScalar = Scalar(0,0,250);
//            if((corner.y<-color.cols*2/2)||(corner.y>color.cols*2))
//                showScalar = Scalar(0,0,250);
//        }

        Mat dst_,src_;
        color.copyTo(src_);
        src_.copyTo(dst_);
        cv::Point pt[1][4];
        pt[0][0] = scene_corners[3];
        pt[0][1] = scene_corners[0];
        pt[0][2] = scene_corners[1];
        pt[0][3] = scene_corners[2];
        const cv::Point* ppt[1]={pt[0]};
        int npt[1] = {4};

        cv::fillPoly(src_,ppt,npt,1,showScalar);
        //cv::rectangle(src,cv::Point(450,100),cv::Point(750,400),cv::Scalar(0,255,0),-1,8);
        cv::addWeighted(dst_,0.7,src_,0.3,0,dst_);
        cv::namedWindow("addweight",cv::WINDOW_NORMAL);
        cv::imshow("addweight", dst_);
//        cv::waitKey();
        last_color = color;


        if(start_flag == 10000)
            return 0;

    }
    return 0;
}

bool Start_Overlap(Mat template_photo){
    H_last_ = Mat::eye(3,3,CV_64FC1);
    H_optflow_ = Mat::eye(3,3,CV_64FC1);
    Max_origin_Size = 300;
    start_flag = false;

    Mat tempPlate;
    resize(template_photo,tempPlate,Size(),0.5,0.5);
    vector<Point2f> empty_l(4);
    scene_corners.swap(empty_l);
    cout << scene_corners<<endl;

    keypoints.clear();
    origin_keypoints.clear();
    // 对第一帧提取FAST特征点
    vector<cv::KeyPoint> kps;
    Ptr<GFTTDetector> gftt = GFTTDetector::create(Max_origin_Size,0.05,
                                                  2,5,true);
    gftt->detect(color,kps);

    for ( auto kp:kps ){
        keypoints.push_back( kp.pt );
        origin_keypoints.push_back( kp.pt);
//        start_keypoints.push_back(kp);
    }

    gftt.release();
    color.copyTo(last_color);
}

bool Calc_Overlap(Mat frame,vector<Point2f>&cornerOutput){



    Mat calc_frame;
    resize(frame,calc_frame,Size(),0.5,0.5);
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    //todo  LK跟踪特征点
    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK( last_color, calc_frame, prev_keypoints, next_keypoints, status, error );

    //todo  把跟丢的点删掉
    int i=0;
    auto iter_tmp = origin_keypoints.begin();
    for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
    {
        if (status[i] == 0 ){
            iter = keypoints.erase(iter);
            iter_tmp = origin_keypoints.erase(iter_tmp);
            continue;
        }
        iter_tmp++;
        *iter = next_keypoints[i];
        iter++;
    }
//    cout << "tracked keypoints: " << keypoints.size()<<endl;
//    cout << "origin_keypoints keypoints: " << origin_keypoints.size()<<endl;
    //todo 对跟踪点转换格式并求Homograph
    for ( auto kp:keypoints )
        cv::circle(color, kp, 2, cv::Scalar(100, 240, 0), 1);
    Mat src_points(1, static_cast<int>(prev_keypoints.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(prev_keypoints.size()), CV_32FC2);
    vector<KeyPoint> srcKeypoint,dstKeypoint;
    //遍历所有匹配点对，得到匹配点对的特征点坐标
    auto psrc =origin_keypoints.begin();
    auto pdst =keypoints.begin();
    for (size_t i = 0; i < prev_keypoints.size(); ++i) {
        src_points.at<Point2f>(0, static_cast<int>(i)) = *psrc;    //特征点坐标赋值
        dst_points.at<Point2f>(0, static_cast<int>(i)) = *pdst;    //特征点坐标赋值
        psrc++;
        pdst++;
    }
    vector<uchar> origin_kp_mask;
    Mat H_temp = findHomography(src_points,dst_points,
                                LMEDS,5,
                                origin_kp_mask,500,0.95);
    H_optflow_ = H_temp*H_last_;
//        cout << H_optflow_ <<endl;

    //todo 如果跟踪点数少于90%,刷新跟踪
    if (keypoints.size() < Max_origin_Size*0.9)
    {
        Trans_H();
        H_optflow_.copyTo(H_last_);
    }

    //todo 如果计算正确,那么可以求出overlap区域的corner投影
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)color.cols, 0 );
    obj_corners[2] = Point2f( (float)color.cols, (float)color.rows );
    obj_corners[3] = Point2f( 0, (float)color.rows );
    perspectiveTransform(obj_corners, scene_corners, H_optflow_);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration <double> time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
    cout<<"algorithm use time："<<time_used.count()<<" seconds.\n"<<endl;
    return true;
}

void Trans_H(Mat cmt){
    keypoints.clear();
    origin_keypoints.clear();
    // 对第一帧提取FAST特征点
    vector<cv::KeyPoint> kps;
    Mat kps_g;
    Ptr<GFTTDetector> gftt = GFTTDetector::create(Max_origin_Size,
                                                  0.05,2,5,true);
    gftt->detect(cmt,kps);
//        KeyPointsFilter::runByImageBorder(kps, color.size(), 2);
    KeyPointsFilter::retainBest(kps, Max_origin_Size);
    for ( auto kp:kps ){
        keypoints.push_back( kp.pt );
        origin_keypoints.push_back( kp.pt);
    }
    last_color = cmt;

}
