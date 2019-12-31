//
// Created by aruchid on 2019/12/20.
//

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

#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;



//当前帧图片
Mat frame, gray,LastPic,src,dst;
//前一帧图片
Mat prev_frame, prev_gray,last_gray;
//保存特征点
vector<Point2f> features;
//初始化特征点
vector<Point2f> inPoints;

//当前帧和前一帧的点
vector<Point2f> fpts[2];
//特征点跟踪标志位
vector<uchar> status;
//误差和
vector<float> err;

vector<KeyPoint> LastPhoto_keypoints;
Mat LastPhoto_descriptors;

vector<KeyPoint> FramePhoto_keypoints;
Mat FramePhoto_descriptors;


vector<Point2f> LastPhoto_points;
vector<Point2f> LastFrame_points;
vector<Point2f> PerFrame_points;



int main(){
    VideoCapture capture(0);
    // 摄像头读取文件开关



    if (capture.isOpened())
    {
        //设拍照flags
        bool flag =false;
        while (capture.read(frame)) {
            //todo 按S 拍照,并且用AKAZE 录当前帧的特征点
            char keyCode = cv::waitKey(30);
            if (keyCode == 's') {
                // 把图片保存起来

                cout << "s get"<< endl;
                //todo set_src_feature copy

                frame.copyTo(src);
                //把图片的图片转为gray
                cvtColor(src, prev_gray, COLOR_BGR2GRAY);
                Ptr<FeatureDetector> detector = AKAZE::create();
                Ptr<DescriptorExtractor> descriptor = AKAZE::create();

                double maxCorners = 10000.0;
                double qualityLevel = 0.01;
                double minDistance = 20.0;
                double blockSize = 5.0;
                double k = 0.04;
                goodFeaturesToTrack(prev_gray, features, maxCorners, qualityLevel, minDistance, Mat(), blockSize, false, k);

                detector->detect ( src,LastPhoto_keypoints );
                descriptor->compute ( src, LastPhoto_keypoints, LastPhoto_descriptors );
                flag = true;
//                for(auto kpot:LastPhoto_keypoints){
//                    //上一帧的点位置
//                    LastFrame_points.push_back(kpot.pt);
//                }
                for(auto ky:features){

                    LastFrame_points.push_back(ky);
                }

                //上一张照片的点位置,持久化
                LastPhoto_points.assign(LastFrame_points.begin(),LastFrame_points.end());
                cout << LastFrame_points.size() << endl;
                cout << LastPhoto_points.size() << endl;


                continue;
            }
            if(flag){
                cout<<"optical "<< endl;
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                calcOpticalFlowPyrLK(prev_gray,gray,LastFrame_points,PerFrame_points,status,err);
                cout<<"calcOpticalFlowPyrLK "<< endl;
                int k = 0;
                for (int i = 0; i < PerFrame_points.size(); i++) {
//                    double dist = abs(LastFrame_points[i].x- PerFrame_points[i].x) + abs(LastFrame_points[i].y - PerFrame_points[i].y);
                    if ( status[i]) {
                        //删除损失的特征点
                        LastPhoto_points[k] = LastPhoto_points[i];
                        PerFrame_points[k++] = PerFrame_points[i];
                    }
                }
                cout << k << endl;
                LastPhoto_points.resize(k);
                PerFrame_points.resize(k);
                swap(PerFrame_points, LastFrame_points);

                for (size_t t = 0; t <LastFrame_points.size(); t++) {
                    line(frame, LastPhoto_points[t], LastFrame_points[t], Scalar(0, 0, 255), 1, 8, 0);
                    circle(frame, LastFrame_points[t], 2, Scalar(0, 255, 0), 2, 8, 0);
                }

            }


            //保存当前帧为前一帧
//            gray.copyTo(prev_gray);
            imshow("1", frame);
//            waitKey(27);
        }
    }
    return 0;
}