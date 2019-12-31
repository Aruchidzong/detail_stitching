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

#include "../include/aruchid_featuresfind.hpp"
// #include "extra.h" // use this if in OpenCV2
using namespace std;
using namespace cv;
using namespace cv::detail;


#define Trans_rate 0.85
#define Max_origin_  300
Mat H_last_ = Mat::eye(3,3,CV_64FC1);
Mat H_optflow_ = Mat::eye(3,3,CV_64FC1);
;

bool stable_flag = true;
bool start_flag = true;
bool lost_flag = true;
bool find_back_tracking = false;

vector<KeyPoint> last_photo_points;
list<cv::Point2f> origin_keypoints; //只是一个list存储所有的初始点
list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
cv::Mat color, last_color;
std::vector<Point2f> scene_corners(4);
//for 找回
vector<KeyPoint> startKeypoints;
Mat start_Descriptor;

float computeReprojError(
        vector<Point2f> pt_org,
        vector<Point2f> pt_prj,
        Mat H_front
);


Size Get_Overlap_Area(vector<Point2f> corners);
bool Calc_Overlap(int &Max_origin_size);
bool Start_Overlap(int &Max_origin_size);
void Trans_H(int &Max_origin_size);
bool Retrieve_Overlap(int Max_origin_size);
bool Auto_shoot(float size,Mat color,Point2f center);

int main( int argc, char** argv )
{
    cv::VideoCapture cap;
//    cap.open("1.mp4");
//    cap.open("2.mp4");
//    cap.open("3.mp4");
    cap.open("4.mp4");
//    cap.open("5.mp4");
//    cap.open("6.mp4");
//    cap.open("7.mp4");
//    cap.open( "s1.mp4");
//    cap.open("s2.mp4");
//    cap.open("9.mp4");
//    cap.open("10.mp4");
//    cap.open("11.mp4");

//    cap.open(0);
    // 判断摄像头是否打开
    if (!cap.isOpened()){
        std::cerr << "Could't open capture" << std::endl;
        return -1;
    }

    int Max_origin_size;
    char keyCode;

    Scalar showScalar = Scalar(0,255,0);

    for(;;)
    {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cap >> color;
        if(color.empty())
            break;


//        rotate(color,color,ROTATE_90_CLOCKWISE);
        double scale_ = 300./color.rows;
        resize(color ,color ,Size() ,scale_,scale_);
        cout<< color.size << endl;


        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration <double> time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
//        cout<<"Calc_Overlap  use time："<<time_used.count()<<" seconds."<<endl;
//        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        keyCode= cv::waitKey(1);
//        //todo 这里是拍照调用的接口
//        if (keyCode == 's')
//            start_flag = true;
        if (start_flag)
        {
            //清楚所有flag
            Start_Overlap(Max_origin_size);
            cout << "startKeypoints.size()" << endl;
            cout << startKeypoints.size() << endl;
            cout << start_Descriptor.size <<endl;
            continue;
        }
        if ( color.data==nullptr)
            continue;
        if(find_back_tracking){
            find_back_tracking = -1*Retrieve_Overlap(0+Max_origin_);
        }
        if( Retrieve_Overlap(Max_origin_size))
            cout << "retrieve " << endl;
//        if(find_back_tracking){
//            find_back_tracking = false;
//        }
        if(Auto_shoot(0.5,color,Pic_centre[0])) {
            find_back_tracking = true;
        }

//        if(lost_flag)
//        Calc_Overlap(Max_origin_size);
//        if(!lost_flag)
//            lost_flag = Retrieve_Overlap();

//        for(auto corner:scene_corners){
//            if((corner.x<-color.rows*2/2)||(corner.x>color.rows*2))
//                showScalar = Scalar(0,0,250);
//            if((corner.y<-color.cols*2/2)||(corner.y>color.cols*2))
//                showScalar = Scalar(0,0,250);
//        }

        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
        cout<<"Calc_Overlap  use time："<<time_used.count()<<" seconds."<<endl;
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


        line(dst_, scene_corners[0], scene_corners[1], Scalar(0, 240, 0), 5);
        line(dst_, scene_corners[1], scene_corners[2], Scalar(0, 240, 0), 5);
        line(dst_, scene_corners[2], scene_corners[3], Scalar(0, 240, 0), 5);
        line(dst_, scene_corners[3], scene_corners[0], Scalar(0, 240, 0), 5);

//        cv::fillPoly(src_,ppt,npt,1,showScalar);
        //cv::rectangle(src,cv::Point(450,100),cv::Point(750,400),cv::Scalar(0,255,0),-1,8);
//        cv::addWeighted(dst_,0.7,src_,0.3,0,dst_);
        cv::namedWindow("addweight",cv::WINDOW_NORMAL);
        cv::imshow("addweight", dst_);
//        cv::waitKey();


        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        chrono::duration <double> time_used2 = chrono::duration_cast <chrono::duration<double>> ( t3 - t1 );
//        cout<<"imshow  use time："<<time_used2.count()<<" seconds."<<endl;


    }

    return 0;
}

bool Start_Overlap(int & Max_origin_size){
    Max_origin_size = (int)Max_origin_;
    H_last_ = Mat::eye(3,3,CV_64FC1);
    H_optflow_ = Mat::eye(3,3,CV_64FC1);
    stable_flag = true;
//    bool start_flag = true;
    lost_flag = true;
    find_back_tracking = false;
    start_flag = false;

    vector<Point2f> empty_l(4);
    scene_corners.swap(empty_l);
    cout << scene_corners<<endl;
    empty_l.clear();

    keypoints.clear();
    origin_keypoints.clear();
    last_photo_points.clear();
    // 对第一帧提取FAST特征点
    vector<cv::KeyPoint> kps;
//    Ptr< FastFeatureDetector> dst_fastdetector =
//            FastFeatureDetector::create (20, true, FastFeatureDetector::TYPE_9_16);
//    Mat gray ;
//    color.convertTo(gray,COLOR_RGB2GRAY);
    Ptr<GFTTDetector> gftt = GFTTDetector::create(Max_origin_size,0.05,
                                                  2,5,true);
    gftt->detect(color,kps);
//    dst_fastdetector->detect(color,kps);

    KeyPointsFilter::retainBest(kps, Max_origin_size);
//    Max_origin_Size = kps.size();
    //            Mat gray;
    //            color.convertTo(gray,COLOR_RGB2GRAY);
    //            goodFeaturesToTrack(gray, kps_g, 5000, 0.01, 10, Mat(), 3, 3, 0, 0.04);
//    detector->detect( color, kps );
//    KeyPointsFilter::runByImageBorder(kps, color.size(), 2);
    for ( auto kp:kps ){
        keypoints.push_back( kp.pt );
        origin_keypoints.push_back( kp.pt);
        startKeypoints.push_back(kp);
    }
//    Ptr<FeatureDetector>
//    Ptr<FeatureDetector> detail_detector = ::ORB::create();
    Ptr<ORB> extractor= ORB::create(
            1500, 1.5f,
            3, 5,
            0, 2, ORB::HARRIS_SCORE,
            31, 20);

    extractor->compute(color, startKeypoints, start_Descriptor);

    gftt.release();
    extractor.release();
    color.copyTo(last_color);
    kps.clear();
    //    color.convertTo(last_color,COLOR_RGB2GRAY);
}

bool Retrieve_Overlap(int Max_origin_size){
    vector<KeyPoint> frame_keypoints;
    Mat frame_descriptor;
    vector<Point2f> empty_l(4);
    scene_corners.swap(empty_l);
    //    cout << "try to retrieve . " << endl;
    //    cout << scene_corners<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<cv::KeyPoint> kps;
    Mat kps_g;
    cv::Ptr<cv::FastFeatureDetector> detector =
            cv::FastFeatureDetector::create(
                    20,
                    true
            );

//    Ptr<GFTTDetector> gftt = GFTTDetector::create(Max_origin_size,0.05,
//                                                  2,5,true);
//    gftt->detect(color,kps);
////    dst_fastdetector->detect(color,kps);
//
//    KeyPointsFilter::retainBest(kps, Max_origin_size);
    //            Mat gray;
    //            color.convertTo(gray,COLOR_RGB2GRAY);
    //            goodFeaturesToTrack(gray, kps_g, 5000, 0.01, 10, Mat(), 3, 3, 0, 0.04);
    detector->detect( color, frame_keypoints );
    //    KeyPointsFilter::runByImageBorder(frame_keypoints, color.size(), 2);
    KeyPointsFilter::retainBest(frame_keypoints, Max_origin_size);

    Ptr<ORB> extractor= ORB::create(
            500, 1.5f,
            3, 5,
            0, 2, ORB::HARRIS_SCORE,
            31, 20);
    extractor->compute(color, frame_keypoints,frame_descriptor);

//    cout<<"frame feature" <<endl;
//    cout << frame_keypoints.size() << endl;
//    cout << frame_descriptor.size << endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration <double> time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );

    if(frame_keypoints.size() < startKeypoints.size() / 2)
        return false;
    //    cout<<"orb "<<frame_keypoints.size()<<" use time："<<time_used.count()<<" seconds."<<endl;
    for ( auto kp:frame_keypoints )
        cv::circle(color, kp.pt, 2,
                   cv::Scalar(100, 240, 0),
                   1);
    vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
//    matcher->match ( descriptors_1, descriptors_2, match );
    std::vector<Point2f> obj_pt, scene_pt;  //重复区域匹配点list
    KnnMatcher(start_Descriptor, frame_descriptor, matches, 0.25);
    for (size_t i = 0; i < matches.size(); ++i) {
        const DMatch &m = matches[i];
        //得到第一幅图像的当前匹配点对的特征点坐标
        Point2f p = startKeypoints[m.queryIdx].pt;
        obj_pt.push_back(p);    //特征点坐标赋值
        p = frame_keypoints[m.trainIdx].pt;
        scene_pt.push_back(p);
    }

//    Ptr<cv::flann::Index> tree;
//    tree.release();
//    tree = makePtr<cv::flann::Index>(start_Descriptor,
//                                     cv::flann::LshIndexParams(
//                                             5,
//                                             15,
//                                             0),
//                                     cvflann::FLANN_DIST_HAMMING
//    );
//
//    //flann knn search
//    cv::Mat indices, dists;
//    Ptr<cv::flann::SearchParams> flann_search_dst =
//            makePtr<flann::SearchParams>(32, 0, false);
//    tree->knnSearch(frame_descriptor,
//                    indices, dists, 2,
//                    *flann_search_dst
//    );
//    cout << "tree" <<endl;
    //get match points
//    obj_pt.clear();
//    scene_pt.clear();
//    float* dists_ptr;
//    int* indeces_ptr;
//    for(int i=0;i<dists.rows;i++)
//    {
//        dists_ptr=dists.ptr<float>(i);
//        indeces_ptr = indices.ptr<int>(i);
//        if (dists_ptr[0]<(1.f - 0.25)*dists_ptr[1])
//        {
//            obj_pt.push_back( startKeypoints[indeces_ptr[0]].pt );
//            scene_pt.push_back( frame_keypoints[i].pt );
//        }
//    }
//    cout<< "obj_pt = "<< obj_pt.size() <<endl;
    if(obj_pt.size()<frame_keypoints.size()/20|| obj_pt.size()<6)
        return false;

//    cout
    std::vector<uchar> inliers_mask;
    cv::Mat H = findHomography(obj_pt, scene_pt,
                               inliers_mask,LMEDS);

    int num_inliers = 0;    //匹配点对的内点数先清零
    //由内点掩码得到内点数
    vector<Point2f>inlier_org,inlier_prj;
    for (size_t i = 0; i < inliers_mask.size(); ++i)    //遍历匹配点对，得到内点
    {
        if (!inliers_mask[i])    //不是内点
            continue;
        Point2f p = obj_pt[i];    //第一幅图像的内点坐标
        inlier_org.push_back(p);
                p = scene_pt[i];    //第一幅图像的内点坐标
        inlier_prj.push_back(p);
        num_inliers++;
//        matches_output->push_back(matches[i]);
    }

    float confidence = num_inliers / (8 + 0.3 * obj_pt.size());
    cout << "confidence" << confidence << endl;
    float err = computeReprojError(inlier_org,inlier_prj,H);
    cout << "err" << err <<endl;
    if (err>5){
        return false;
    }
    if(confidence<0.8)
        return false;
    Point2f centerPt ;


//    else if (confidence>2.
//    }
    std::vector<Point2f> obj_corners(4);
    std::vector<Point2f> temp_scenc_cor(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)color.cols, 0 );
    obj_corners[2] = Point2f( (float)color.cols, (float)color.rows );
    obj_corners[3] = Point2f( 0, (float)color.rows );
    perspectiveTransform(obj_corners, temp_scenc_cor, H);
    for (int i = 0; i < 4; ++i) {
        centerPt.x += temp_scenc_cor[i].x;
        centerPt.y += temp_scenc_cor[i].y;
    }
    centerPt = centerPt/4;
    if(centerPt.x>(color.cols/2+color.cols/10))
        return false;
    if(centerPt.x<(color.cols/2-color.cols/10))
        return false;
    if(centerPt.y>(color.rows/2+color.rows/10))
        return false;
    if(centerPt.y<(color.rows/2-color.rows/10))
        return false;
    temp_scenc_cor.swap(scene_corners);
    time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
//    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    Size Sizeof_overlap = Get_Overlap_Area(scene_corners);


//    int size_ovl;
//    if(color.cols>color.rows)
//        size_ovl =
//                Sizeof_overlap.height +
//                (color.rows/color.cols)*Sizeof_overlap.width;
//    else if (color.rows>color.cols)
//        size_ovl =
//                Sizeof_overlap.width +
//                (color.cols/color.rows)*Sizeof_overlap.height;
////    cout << "size_ovl" << endl;
////    cout << size_ovl << endl;
////    cout << 1*min(color.cols,color.rows) << endl;
//
//    if(
//            size_ovl < 0.9*min(color.cols,color.rows)||
//            size_ovl > 1.2*min(color.cols,color.rows)
//    ){
//        cout << false;
//        return false;
//    }

    lost_flag = true;
    H_last_ = H;
    return true;
}

bool Calc_Overlap(int &Max_origin_size){
    // 对其他帧用LK跟踪特征点
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;

    cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
//        cout << prev_keypoints.size() << " =? " << next_keypoints.size() << endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration <double> time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    // 把跟丢的点删掉
    int i=0;int number_stable = 0;
    auto iter_tmp = origin_keypoints.begin();
    for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
    {
        double dist = abs(prev_keypoints[i].x- next_keypoints[i].x) + abs(prev_keypoints[i].y - next_keypoints[i].y);
//        if(dist<2)
//            number_stable++;
        if (status[i] == 0 )
//            if (status[i] == 0 )
        {

            iter = keypoints.erase(iter);
            iter_tmp = origin_keypoints.erase(iter_tmp);
            continue;
        }
        iter_tmp++;
        *iter = next_keypoints[i];
        iter++;
    }

    cout << "tracked keypoints: " << keypoints.size()<<endl;
    cout << "origin_keypoints keypoints: " << origin_keypoints.size()<<endl;
//    cout << "number_stable" << number_stable << endl;
    //说明相机稳定
//    if (number_stable > keypoints.size()*2/4)
//    {
//        keypoints.clear();
//        origin_keypoints.clear();
//        // 对第一帧提取FAST特征点
//        vector<cv::KeyPoint> kps;
//        Mat kps_g;
//        cv::Ptr<cv::FastFeatureDetector> detector =
//                cv::FastFeatureDetector::create(10, true,
//                                                FastFeatureDetector::TYPE_9_16);
//        //            Mat gray;
//        //            color.convertTo(gray,COLOR_RGB2GRAY);
//        //            goodFeaturesToTrack(gray, kps_g, 5000, 0.01, 10, Mat(), 3, 3, 0, 0.04);
//        detector->detect( color, kps );
////        KeyPointsFilter::runByImageBorder(kps, color.size(), 2);
//        KeyPointsFilter::retainBest(kps, Max_origin_Size);
//        for ( auto kp:kps ){
//            keypoints.push_back( kp.pt );
//            origin_keypoints.push_back( kp.pt);
//        }
//        last_color = color;
//        H_optflow_.copyTo(H_last_);
//        cout<<"most keypoints are stable."<<endl;
//        stable_flag = true;
////        cout << "stable++++++"<< endl;
//    }

    // 画出 keypoints
//    cv::Mat col = color.clone();
    for ( auto kp:keypoints )
        cv::circle(color, kp, 2, cv::Scalar(0, 0, 200), 1);

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
                                origin_kp_mask,1000,0.99);

    H_optflow_ = H_temp*H_last_;

//        cout << H_optflow_ <<endl;


    //todo 如果计算正确,那么可以求出overlap区域的corner投影

    if (keypoints.size() < Max_origin_size*Trans_rate)
    {
        Trans_H(Max_origin_size);
//        find_back_tracking = true;
        H_optflow_.copyTo(H_last_);
    }
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)color.cols, 0 );
    obj_corners[2] = Point2f( (float)color.cols, (float)color.rows );
    obj_corners[3] = Point2f( 0, (float)color.rows );
    perspectiveTransform(obj_corners, scene_corners, H_optflow_);
    time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );

    cout<<"algorithm use time："<<time_used.count()<<" seconds.\n"<<endl;


    Size Sizeof_overlap = Get_Overlap_Area(scene_corners);

    int size_ovl;
    if(color.cols>color.rows)
        size_ovl =
                Sizeof_overlap.height +
                (color.rows/color.cols)*Sizeof_overlap.width;
    else if (color.rows>color.cols)
        size_ovl =
                Sizeof_overlap.width +
                (color.cols/color.rows)*Sizeof_overlap.height;
    cout << "size_ovl" << endl;
    cout << size_ovl << endl;
    cout << 1*min(color.cols,color.rows) << endl;

    if(size_ovl < 1*min(color.cols,color.rows)){
        cout << false;
        return false;
    }

    last_color = color;
    return true;
}


bool Auto_shoot(float size,Mat colr_,Point2f center){
//    cout << colr_.cols/2 <<" "<< colr_.rows/2 <<endl;
//    cout << center.x <<" " <<center.y <<endl;
//    cout << (colr_.cols/2)*size <<endl;
//    cout << (((colr_.cols/2)*(1-size))+colr_.cols/2) <<endl;
//    cout << (colr_.rows/2)*size <<endl;
//    cout << (((colr_.rows/2)*(1-size))+colr_.rows/2) <<endl;

    if(center.x < ((colr_.cols/2)*size))
        return false;
    if(center.x > (((colr_.cols/2)*(1-size))+colr_.cols/2))
        return false;
    if(center.y < (((colr_.rows/2)*size)))
        return false;
    if(center.y > (((colr_.rows/2)*(1-size))+colr_.rows/2))
        return false;
    return true;
}

bool Check_Overlap_Wrong(
        Mat color,
        vector<uchar>inliers_mask,
        vector<Point> obj_pt,
        vector<Point> scene_corners
){

    int num_inliers = 0;    //匹配点对的内点数先清零
    //由内点掩码得到内点数
    for (size_t i = 0; i < inliers_mask.size(); ++i)
        if (inliers_mask[i])
            num_inliers++;

    float confidence = num_inliers / (8 + 0.3 * obj_pt.size());
    if (confidence < 1.5)
        return true;

    //todo  kpts postion
    for(auto corner:scene_corners){
        if((corner.x<-color.rows*2/2-100)||(corner.x>color.rows*2+100))
            return true;
        if((corner.y<-color.cols*2/2-100)||(corner.y>color.cols*2+100))
            return true;
    }
}

void Trans_H(int &Max_origin_size){
    cout << "trans_HHHHH" <<endl;
    keypoints.clear();
    origin_keypoints.clear();
    // 对第一帧提取FAST特征点
    vector<cv::KeyPoint> kps;
    Mat kps_g;
//    Mat gray ;
//    color.convertTo(gray,COLOR_RGB2GRAY);
    Ptr<GFTTDetector> gftt = GFTTDetector::create(Max_origin_,
                                                  0.05,5,5,true);
    gftt->detect(color,kps);
    Max_origin_size = kps.size();
//    Ptr< FastFeatureDetector> dst_fastdetector =
//            FastFeatureDetector::create (20, true, FastFeatureDetector::TYPE_9_16);

//    Ptr<GFTTDetector> gftt = GFTTDetector::create(Max_origin_Size,0.05,
//            2,5,true);
//    dst_fastdetector->detect(color,kps);

//    KeyPointsFilter::runByImageBorder(kps, color.size(), 2);
//    KeyPointsFilter::retainBest(kps, Max_origin_Size);
//    KeyPointsFilter::retainBest(kps, Max_origin_Size);
    for ( auto kp:kps ){
        keypoints.push_back( kp.pt );
        origin_keypoints.push_back( kp.pt);
    }
    last_color = color;

}

Size Get_Overlap_Area(vector<Point2f> corners){
    // 这里需要改成四个顶点的外接矩形坐标
    Point tmp0, tmp1, LT_position,RB_position;
    //LT_position 是该图片在全景图中的左上角位置坐标
    tmp0.x = min(corners[0].x, corners[1].x);
    tmp1.x = min(corners[2].x, corners[3].x);
    LT_position.x = min(tmp0.x, tmp1.x);

    tmp0.y = min(corners[0].y, corners[1].y);
    tmp1.y = min(corners[2].y, corners[3].y);
    LT_position.y = min(tmp0.y, tmp1.y);

    tmp0.x = max(corners[0].x, corners[1].x);
    tmp1.x = max(corners[2].x, corners[3].x);
    RB_position.x = max(tmp0.x, tmp1.x);

    tmp0.y = max(corners[0].y, corners[1].y);
    tmp1.y = max(corners[2].y, corners[3].y);
    RB_position.y = max(tmp0.y, tmp1.y);
    Size returnout(-LT_position.x+RB_position.x,
                   -LT_position.y+RB_position.y);
    return returnout;
}

float computeReprojError(
        vector<Point2f> pt_org,
        vector<Point2f> pt_prj,
        Mat H_front
        )
//m1和m2为匹配点对
//model表示单应矩阵H
//_err表示所有匹配点对的重映射误差，即式21几何距离的平方
{
    vector<float> err;
    int count = pt_org.size();
    const double* H = reinterpret_cast<const double*>(H_front.data);
    for(int i = 0; i < count; i++ )    //遍历所有特征点，计算重映射误差
    {
        double ww = 1./(H[6]*pt_org[i].x + H[7]*pt_org[i].y + 1.);    //式21中分式的分母部分
        double dx = (H[0]*pt_org[i].x + H[1]*pt_org[i].y + H[2])*ww - pt_prj[i].x;    //式21中x坐标之差
        double dy = (H[3]*pt_org[i].x + H[4]*pt_org[i].y + H[5])*ww - pt_prj[i].y;    //式21中y坐标之差
        err.push_back((float)dx*dx + dy*dy);    //得到当前匹配点对的重映射误差
    }
    double err_mean =  accumulate(err.begin(),err.end(),0.0)/err.size();
    return (float)err_mean;
}