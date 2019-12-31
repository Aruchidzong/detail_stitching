
#include "optical.hpp"

#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
//#include "str_common.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/util.hpp"

#include <numeric>
#include <time.h> //zt
#include <sys/time.h> //zt


#include <list>
#include <vector>
#include <opencv2/video/tracking.hpp>
#include <iomanip>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

//#include "../src/aruchid_fea"

using namespace std;
using namespace cv;
using namespace cv::detail;

#define Rsize_scale 0.25
#define Trans_rate 0.75
#define Max_origin_  200
#define Detail_Max_  100
#define Resize_gool 300.
Mat H_last_    = Mat::eye(3,3,CV_64FC1);
Mat H_optflow_ = Mat::eye(3,3,CV_64FC1);


bool find_back_tracking = false;

list<cv::Point2f> origin_keypoints; //只是一个list存储所有的初始点
list<cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list

vector<KeyPoint> startKeypoints;
Mat start_Descriptor;
cv::Mat templateImg;
int Max_origin_size = Max_origin_;

typedef std::set<std::pair<int,int> > MatchesSet;

void KnnMatcher(Mat descriptors_1,Mat descriptors_2,vector<DMatch>& Dmatchinfo,float match_conf_);
float reprojError(vector<Point2f> pt_org, vector<Point2f> pt_prj, const Mat& H_front);
Size Get_Overlap_Area(vector<Point2f> corners);
bool Calc_Overlap(Mat& dst,int &maxOriginSize);
bool Start_Overlap(int &Max_origin_size);
void updateKeyPoint(const Mat& frame,int &MOS);
bool Retrieve_Overlap(Mat &dst,int Max_origin_Size);
bool Check_rads_bias(float size,const Mat& pic_,const Point2f& center);
void Overlap_calculate_ket(int& Max_origin_size);
void cal_area_corner(const Mat& pic,const Mat& H_final, vector<Point>&);
bool Check_center_crossborder(const Mat& pic,float border_rate,Mat H);
bool Check_Area_Correct(vector<Point>dst_pnts,cv::Mat &dst);

inline float getSideLength(const Point2f &p1, const Point2f &p2);
inline float getSideVec(const Point2f &p1, const Point2f &p2);
inline Mat resize_input(Mat &frame);
inline void resize_output(vector<Point>& dst_pt);

inline float getSideLength(const Point2f &p1, const Point2f &p2)
{
    float sideLength = sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
    return sideLength;
}
inline float getSideVec(const Point2f &p1, const Point2f &p2, const Point2f & p3)
{
    float sideVec = (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y);
    return sideVec;
}
bool Check_Area_Correct(vector<Point2f>dst_pnts,cv::Mat &dst);


Mat resize_input(Mat &frame){
    Mat output;
    float scale = Rsize_scale;
//    scale = Resize_gool/frame.cols;
    resize(frame,output,Size(),scale,scale,INTER_LINEAR);
    return output;
}
void resize_output(vector<Point>& dst_pt){
    for (int i = 0; i < 4; ++i) {
        dst_pt[i] =dst_pt[i]/Rsize_scale;
    }
}
bool Check_Area_Correct(vector<Point>dst_pnts,cv::Mat &dst){

    //判断重叠区域是否为四边形
    if(dst_pnts.size() != 4 || dst_pnts[0].x >= dst_pnts[1].x || dst_pnts[3].x >= dst_pnts[2].x || dst_pnts[0].y >= dst_pnts[3].y || dst_pnts[1].y >= dst_pnts[2].y)
    {
        std::cout << "OverLap: overlap area is not Quadrilateral. " << std::endl;
        return false;
    }

    //判断四边形内角
    float topSide = getSideLength(dst_pnts[0], dst_pnts[1]); //四边形的上边
    float bottomSide = getSideLength(dst_pnts[2], dst_pnts[3]); //四边形的下边
    float leftSide = getSideLength(dst_pnts[0], dst_pnts[3]); //四边形的左边
    float rightSide = getSideLength(dst_pnts[1], dst_pnts[2]); //四边形的右边
    std::cout << "topSide = " << topSide << ", bottomSide = " << bottomSide << ", leftSide = " << leftSide << ", rightSide = " << rightSide << std::endl;


    float lefttopVec = getSideVec(dst_pnts[0], dst_pnts[1], dst_pnts[3]);
    float righttopVec = getSideVec(dst_pnts[1], dst_pnts[0], dst_pnts[2]);
    float rightbottomVec = getSideVec(dst_pnts[2], dst_pnts[1], dst_pnts[3]);
    float leftbottomVec = getSideVec(dst_pnts[3], dst_pnts[0], dst_pnts[2]);


    int angle0 = acos(lefttopVec / (topSide * leftSide)) * 180 / 3.1416 + 0.5;
    int angle1 = acos(righttopVec / (topSide * rightSide)) * 180 / 3.1416 + 0.5;
    int angle2 = acos(leftbottomVec / (bottomSide * leftSide)) * 180 / 3.1416 + 0.5;
    int angle3 = acos(rightbottomVec / (bottomSide * rightSide)) * 180 / 3.1416 + 0.5;

    std::cout << "angle0 = " << angle0 << ", angle1 = " << angle1 << ", angle2 = " << angle2 << ", angle3 = " << angle3 << std::endl;
    if(abs(angle0) < 30 || abs(angle1) < 30 || abs(angle2) < 30 || abs(angle3) < 30)
    {
        //else
        std::cout << "OverLap: inline angle < 30. " << std::endl;
        return false;
    }
    return true;


}

bool overlap_point(cv::Mat &dst,std::vector<cv::Point> &src_pnts,std::vector<cv::Point> &dst_pnts){
    //找回机制
    dst = resize_input(dst);
    if(find_back_tracking){
        cout<<"find_back_tracking" <<endl;
        if(Retrieve_Overlap(dst,Max_origin_))
            find_back_tracking  = !find_back_tracking;
        if (!find_back_tracking){
            Max_origin_size = Max_origin_;
            updateKeyPoint(dst,Max_origin_size);
        }
    }
    else{
        if(!Calc_Overlap(dst,Max_origin_size))
            find_back_tracking  = true;
    }


    //测量中心点的偏移
    vector<Point2f> Pic_centre;
    Pic_centre.emplace_back((float)(dst.cols/2),(float)(dst.rows/2));
    perspectiveTransform(Pic_centre,Pic_centre,H_optflow_);
    if(!Check_rads_bias(-0.2,dst,Pic_centre[0])) {
        find_back_tracking = true;
        return false;
    }
    vector<Point> dst_(4);
    cal_area_corner(dst,H_optflow_,dst_pnts);
    cout << dst_pnts <<endl;
    if(!Check_Area_Correct(dst_pnts,dst))
        find_back_tracking = true;
    if (find_back_tracking)
        return false;
    resize_output(dst_pnts);
    for(const auto& pp:keypoints)
        circle(dst,pp,2,Scalar(0,200,0),2);
    return true;
}
bool set_src_feature( cv::Mat & frame){


    H_last_ = Mat::eye(3,3,CV_64FC1);
    H_optflow_ = Mat::eye(3,3,CV_64FC1);
    find_back_tracking = false;
    origin_keypoints.clear(); //只是一个list存储所有的初始点
    keypoints.clear();      // 因为要删除跟踪失败的点，使用list
    startKeypoints.clear();
    start_Descriptor.release();
    templateImg.release();
    Max_origin_size = Max_origin_;

    frame = resize_input(frame);
    // 对第一帧提取FAST特征点
    vector<cv::KeyPoint> kps;
    Ptr<GFTTDetector> gftt = GFTTDetector::create(Max_origin_);
    gftt->detect(frame,kps);
    //    KeyPointsFilter::retainBest(kps, Max_origin_);
    for ( const auto& kp:kps ){
        keypoints.push_back( kp.pt );
        origin_keypoints.push_back( kp.pt);
    }

    cout  << "keypoints? = " << keypoints.size() <<endl;
    Ptr<xfeatures2d::SIFT> detail_detector = xfeatures2d::SIFT::create(Max_origin_,2);
    //    Ptr<DescriptorExtractor> detail_extractor= xfeatures2d::SIFT::create(Max_origin_);
    detail_detector->detectAndCompute(frame, Mat(), startKeypoints, start_Descriptor);
    //    cout << start_Descriptor.size <<  "? = " << startKeypoints.size() <<endl;
    gftt.release();
    detail_detector.release();
    //    detail_extractor.release();
    kps.clear();
    frame.copyTo(templateImg);

    return true;
}

bool Retrieve_Overlap(Mat& dst,int Max_origin_Size){
    vector<KeyPoint> frame_keypoints;
    Mat frame_descriptor;

    //先不断地去寻找 SIFT特征匹配
    Ptr<xfeatures2d::SIFT> detail_detector = xfeatures2d::SIFT::create(Detail_Max_,2);
    detail_detector->detectAndCompute(dst,Mat(),frame_keypoints,frame_descriptor);


    if(frame_keypoints.size() < startKeypoints.size() / 20)
        return false;
    vector<DMatch> matches;
    std::vector<Point2f> start_pt, frame_pt;  //重复区域匹配点list
    KnnMatcher(start_Descriptor, frame_descriptor, matches, 0.25);
    //将kNN的配对填入两个缓冲 vector
    for (auto m : matches) {
        //得到第一幅图像的当前匹配点对的特征点坐标
        Point2f p = startKeypoints[m.queryIdx].pt;
        start_pt.push_back(p);    //特征点坐标赋值
        p = frame_keypoints[m.trainIdx].pt;
        frame_pt.push_back(p);
    }
    if( start_pt.size()<6||frame_pt.size()<6)
        return false;

    std::vector<uchar> inliers_mask;
    cv::Mat H = findHomography(start_pt, frame_pt,inliers_mask,LMEDS);
    int num_inliers = 0;    //匹配点对的内点数先清零
    //由内点掩码得到内点数
    //在KNN基础上再得到内点 vector
    vector<Point2f>inlier_org,inlier_prj;
    for (size_t i = 0; i < inliers_mask.size(); ++i)    //遍历匹配点对，得到内点
    {
        if (!inliers_mask[i])    //不是内点
            continue;
        Point2f p = start_pt[i];    //第一幅图像的内点坐标
        inlier_org.push_back(p);
        p = frame_pt[i];    //第一幅图像的内点坐标
        inlier_prj.push_back(p);
        num_inliers++;
    }

    float confidence = num_inliers / (8 + 0.3 * start_pt.size());
//    cout << "confidence = " << confidence << endl;
    float err = reprojError(inlier_org, inlier_prj, H);
//    cout << "err = " << err <<endl;
    if (err>5){
        return false;
    }
    if(confidence<1.2)
        return false;
    if(!Check_center_crossborder(dst,0.1,H.inv())){
//        cout << "check false;" <<endl;
        return false;
    }
    //再次更新 两个H矩阵
    H.copyTo(H_optflow_);
    H.copyTo(H_last_);

    start_pt.clear();
    frame_pt.clear();
    detail_detector.release();
    frame_descriptor.release();
//    cout << "auto_retrieve" <<endl;
    return true;
}
void cal_area_corner(const Mat& pic,const Mat& H_final,vector<Point>& output_pt){
    std::vector<Point2f> template_pt(4);
    std::vector<Point2f> output_temp_corner(4);
    template_pt[0] = Point2f(0, 0);
    template_pt[1] = Point2f((float)pic.cols, 0 );
    template_pt[2] = Point2f((float)pic.cols, (float)pic.rows );
    template_pt[3] = Point2f( 0, (float)pic.rows );
    perspectiveTransform(template_pt, output_temp_corner, H_final);
    for (int j = 0; j < 4; ++j) {
        output_pt[j] = Point((int)output_temp_corner[j].x,(int)output_temp_corner[j].y);
    }
}
void KnnMatcher(const Mat descriptors_1,const Mat descriptors_2,vector<DMatch>& Dmatchinfo,const float match_conf_) {
    Dmatchinfo.clear();    //清空
    //定义K-D树形式的索引
    Ptr<flann::IndexParams> indexParams = new flann::KDTreeIndexParams();
    //定义搜索参数
    Ptr<flann::SearchParams> searchParams = new flann::SearchParams();

    if (descriptors_1.depth() == CV_8U) {
        indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
        searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
    }
    //使用FLANN方法匹配，定义matcher变量
    FlannBasedMatcher matcher(indexParams, searchParams);
    vector<vector<DMatch> > pair_matches;    //表示邻域特征点
    MatchesSet matches;    //表示匹配点对

    // Find 1->2 matches
    //在第二幅图像中，找到与第一幅图像的特征点最相近的两个特征点
    matcher.knnMatch(descriptors_1, descriptors_2, pair_matches, 2);
    for (auto & pair_matche : pair_matches)    //遍历这两次匹配结果
    {
        //如果相近的特征点少于2个，则继续下个匹配
        if (pair_matche.size() < 2)
            continue;
        //得到两个最相近的特征点
        const DMatch &m0 = pair_matche[0];
        const DMatch &m1 = pair_matche[1];
        //比较这两个最相近的特征点的相似程度，当满足一定条件时（用match_conf_变量来衡量），才能认为匹配成功
        //TODO match_conf_越大说明第二相似的匹配点，要与第一相似的匹配点差距越大，也就是匹配的要求
        // 特例性，match_conf_ 越大 粗匹配点数越少
        if (m0.distance < (1.f - match_conf_) * m1.distance)    //式1
        {
            //把匹配点对分别保存在matches_info和matches中
            Dmatchinfo.push_back(m0);
            matches.insert(make_pair(m0.queryIdx, m0.trainIdx));
        }
    }

    pair_matches.clear();    //变量清零
}
bool Check_center_crossborder(const Mat& pic,float border_rate,Mat H){
    vector<Point2f> Pic_centre;
    Pic_centre.emplace_back((pic.cols/2),(pic.rows/2));
    perspectiveTransform(Pic_centre,Pic_centre,H);
    return Check_rads_bias(border_rate, pic, Pic_centre[0]);
}
bool Calc_Overlap(Mat &dst,int &maxOriginSize){
    // 对其他帧用LK跟踪特征点
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    //填入 buffer
    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( const auto& kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
//    cout << prev_keypoints.size/**/

    //计算光流跟踪点
    cv::calcOpticalFlowPyrLK( templateImg, dst, prev_keypoints, next_keypoints, status, error );
//        cout << prev_keypoints.size() << " =? " << next_keypoints.size() << endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration <double> time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    // 把跟丢的点删掉
    int i=0;int number_stable = 0;
    auto iter_tmp = origin_keypoints.begin();
    for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
    {
        if (status[i] == 0 ){
            //同步删除那些origin_keypoints 中的对应点
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
    //如果光流跟踪到的点特别少直接退出
    if(keypoints.size()<Max_origin_/10)
        return false;

    Mat src_points(1, static_cast<int>(prev_keypoints.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(prev_keypoints.size()), CV_32FC2);
    vector<KeyPoint> srcKeypoint,dstKeypoint;
    //遍历所有匹配点对，得到匹配点对的特征点坐标
    auto psrc =origin_keypoints.begin();
    auto pdst =keypoints.begin();
    for (size_t step = 0; step < prev_keypoints.size(); ++step) {
        src_points.at<Point2f>(0, static_cast<int>(step)) = *psrc;    //特征点坐标赋值
        dst_points.at<Point2f>(0, static_cast<int>(step)) = *pdst;    //特征点坐标赋值
        psrc++;
        pdst++;
    }
    vector<uchar> origin_kp_mask;
    Mat H_temp = findHomography(src_points,dst_points,LMEDS,5,
                                origin_kp_mask,1000,0.99);
    H_optflow_ = H_temp*H_last_;
    //    cout << H_optflow_ <<endl;
    //todo 如果光流跟踪点数量少于0.9
//    if(Max_origin_size < Max_origin_/3)
//        return false;
    if (keypoints.size() < maxOriginSize * Trans_rate)
    {
//        return false;
        updateKeyPoint(dst, maxOriginSize);
        H_optflow_.copyTo(H_last_);
    }
    dst.copyTo(templateImg);

    time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
    cout<<"algorithm use time："<<time_used.count()<<" seconds.\n"<<endl;
    return true;
}
void updateKeyPoint(const Mat& frame,int &MOS){
    //先清空 vector point buffer
    keypoints.clear();
    origin_keypoints.clear();
    vector<cv::KeyPoint> kps;
    Ptr<GFTTDetector> gftt =
            GFTTDetector::create(Max_origin_,
                    0.01,
                    1,
                    3,
                    true);
    gftt->detect(frame,kps);
    MOS = kps.size();

    for ( const auto& kp:kps ){
        keypoints.push_back( kp.pt );
        origin_keypoints.push_back( kp.pt);
    }
    frame.copyTo(templateImg);
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
bool Check_rads_bias(float size,const Mat& pic_,const Point2f& center){

    float rad_least = min(pic_.rows/2,pic_.cols/2);
    float centre_x = center.x - pic_.cols/2;
    float centre_y = center.y - pic_.rows/2;
//    cout << centre_x << " " <<centre_y << endl;
//    float centre_y;
    float rad_dis = sqrtf(centre_x*centre_x+centre_y*centre_y);
//    cout << "rand_dis" << rad_dis <<endl;
    return (rad_dis < rad_least*(1-size));

}
float reprojError(vector<Point2f> pt_org, vector<Point2f> pt_prj, const Mat& H_front)
{
    //m1和m2为匹配点对
//model表示单应矩阵H
//_err表示所有匹配点对的重映射误差，即式21几何距离的平方
    Mat H_inerr;
    H_front.copyTo(H_inerr);
//    cout << "H_inerr : " <<H_inerr <<endl;
    vector<double> err;
    err.clear();
//    int count = pt_org.size();
//    const double* H = reinterpret_cast<const double*>(H_front.data);
    for(int i = 0; i < pt_org.size(); i++ )    //遍历所有特征点，计算重映射误差
    {
        double ww = 1./(H_inerr.at<double>(2,0)*pt_org[i].x + H_inerr.at<double>(2,1)*pt_org[i].y + 1.);    //式21中分式的分母部分
        double dx = (H_inerr.at<double>(0,0)*pt_org[i].x +
                     H_inerr.at<double>(0,1)*pt_org[i].y +
                     H_inerr.at<double>(0,2))*
                    ww - pt_prj[i].x;    //式21中x坐标之差
        double dy = (H_inerr.at<double>(1,0)*pt_org[i].x +
                     H_inerr.at<double>(1,1)*pt_org[i].y +
                     H_inerr.at<double>(1,2))*
                    ww - pt_prj[i].y;    //式21中y坐标之差
        err.push_back((double)dx*dx + (double)dy*dy);    //得到当前匹配点对的重映射误差
    }
    double err_mean =  (accumulate(err.begin(),err.end(),0.0))/(double)pt_org.size() ;
    return (float)err_mean;
}