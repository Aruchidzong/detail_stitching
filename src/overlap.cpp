//
// Created by aruchid on 2019/12/7.
//



#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "../include/overlap.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;
#include "iostream"
#include <sys/time.h>


//RNG rng;
//1.kalman filter setup

list<float> confidence_List(10);
vector<Point> corner_extern(4);
list<Point2d> move_extern(4);

bool flat_false =false;



bool set_src_feature(cv::Mat &src, ImageFeatures &src_features){
//    move_extern.swap(move_extern_temp);
    list<Point2d> move_extern_temp(8,Point2d(0,0));
    move_extern.swap(move_extern_temp);

    for(auto mv:move_extern){
        cout << mv;
    }
    if (src.empty())
    {
        std::cout << "src is empty." << std::endl;
        return false;
    }
    double overlap_input_megapix = 1;  //输入图片默认尺寸
    double overlap_input_scale = 0;
    int overlap_work_pixel = 0;
    int maxkeypoint = 5000;  //重复区域最大匹配点数
    Mat overlap_input;
    //resize input image
    //if (!is_overlap_scale_set) {
    if (overlap_work_pixel == 0){
        overlap_input_scale = min(1.0, sqrt(overlap_input_megapix * 1e6 / src.size().area()));
        //is_overlap_scale_set = true;
    }
    else{
        if (min(src.rows, src.cols)<= overlap_work_pixel){
            overlap_input_scale = 1;
            //is_overlap_scale_set = true;
        }
        else{
            overlap_input_scale = overlap_work_pixel*1./min(src.rows, src.cols);
            //is_overlap_scale_set = true;
        }
    }
    resize(src, overlap_input, Size(), overlap_input_scale, overlap_input_scale, INTER_NEAREST);

    cv::Mat gray;
    if(overlap_input.channels() == 3)
        cv::cvtColor(overlap_input, gray, COLOR_BGR2GRAY);
    else
        overlap_input.copyTo(gray);

    //feature detect
    // Ptr<DescriptorExtractor> src_extractor;  //重复区域特征描述指针
    struct timeval detect_start, detect_end;
    gettimeofday( &detect_start, NULL );
    Ptr<FastFeatureDetector> src_fastdetector = FastFeatureDetector::create (20, true, FastFeatureDetector::TYPE_9_16);  //重复区域fast特征计算指针
    src_fastdetector->detect(gray, src_features.keypoints);

//    gettimeofday( &detect_end, NULL );
    //求出两次时间的差值，单位为us
//    int detecttimeuse = 1000000 * (detect_end.tv_sec - detect_start.tv_sec ) + detect_end.tv_usec - detect_start.tv_usec;
//    std::cout << "detect time is " << detecttimeuse << "  us."<< std::endl;

    if (src_features.keypoints.size() < 200)
    {
        src_fastdetector.release();
        std::cout << "src feature key size < 200. " << std::endl;
        return false;
    }

    KeyPointsFilter::runByImageBorder(src_features.keypoints, gray.size(), 31);
    KeyPointsFilter::retainBest(src_features.keypoints, maxkeypoint);

    struct timeval description_start, description_end;
    gettimeofday( &description_start, NULL );
    //feature description
    Ptr<ORB> src_extractor = ORB::create(10000, 1.5f, 1, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    src_extractor->compute(gray, src_features.keypoints, src_features.descriptors);
    src_features.descriptors = src_features.descriptors.reshape(1, (int)src_features.keypoints.size());

    gettimeofday( &description_end, NULL );
    //求出两次时间的差值，单位为us
    int descriptiontimeuse = 1000000 * ( description_end.tv_sec - description_start.tv_sec ) + description_end.tv_usec - description_start.tv_usec;
    std::cout << "description time is " << descriptiontimeuse << "  us."<< std::endl;
    std::cout << "src_features descriptors size = " << src_features.descriptors.size()<< std::endl;
    //Ptr<cv::flann::Index> tree_src;  //重复区域flann索引指针

    struct timeval first_find_start, first_find_end;
    gettimeofday( &first_find_start, NULL );

    gettimeofday( &first_find_end, NULL );
    //求出两次时间的差值，单位为us
    int firstfindtimeuse = 1000000 * ( first_find_end.tv_sec - first_find_start.tv_sec ) + first_find_end.tv_usec - first_find_start.tv_usec;
    std::cout << "first img find time is " << firstfindtimeuse << "  us."<< std::endl;

    src_extractor.release();
    src_fastdetector.release();
    return true;
}


bool overlap_point(
        cv::Mat &dst,
        ImageFeatures &src_features ,
        const Point3d target_Angle,
        const Point3d Eular_Angle,
        vector<Point> &src_pnts,
        vector<Point> &dst_pnts){


    double app_start_time = getTickCount();
    if (dst.empty())
    {
        std::cout << "dst is empty. " << std::endl;
        return false;
    }

    if(src_features.keypoints.size() == 0)
    {
        std::cout << "src_features kpt size = 0." << std::endl;
        return false;
    }

    double overlap_input_megapix = 1;  //输入图片默认尺寸
    double overlap_input_scale = 0.0;
    int overlap_work_pixel = 0;
    int maxkeypoint = 5000;

    Mat overlap_input;
    //resize input image
    //if (!is_overlap_scale_set) {
    if (overlap_work_pixel == 0){
        overlap_input_scale = min(1.0, sqrt(overlap_input_megapix * 1e6 / dst.size().area()));
        //is_overlap_scale_set = true;
    }
    else{
        if (min(dst.rows, dst.cols)<= overlap_work_pixel){
            overlap_input_scale = 1;
            //is_overlap_scale_set = true;
        }
        else{
            overlap_input_scale = overlap_work_pixel*1./min(dst.rows, dst.cols);
            //is_overlap_scale_set = true;
        }
    }
    // }
    if(dst.channels() == 3)
        cv::cvtColor(dst, overlap_input, COLOR_BGR2GRAY);
    else
        dst.copyTo(overlap_input);
    resize(overlap_input, overlap_input, Size(), overlap_input_scale, overlap_input_scale, INTER_NEAREST);

    //    cv::Mat gray;
    //    cv::cvtColor(input, gray, CV_BGR2GRAY);

    //    cv::medianBlur(input, input, 3);
    //    cv::GaussianBlur(input, input, cv::Size(3,3), 0.5, 0.5);
    //feature detect
    ImageFeatures dst_features;
    Ptr< FastFeatureDetector> dst_fastdetector = FastFeatureDetector::create (20, true, FastFeatureDetector::TYPE_9_16);
    dst_fastdetector->detect(overlap_input, dst_features.keypoints);

    if (dst_features.keypoints.size() < 150)
    {
        dst_fastdetector.release();
        std::cout << "dst feature kp size < 200. " << std::endl;
        return false;
    }

    KeyPointsFilter::runByImageBorder(dst_features.keypoints, overlap_input.size(), 50);
    KeyPointsFilter::retainBest(dst_features.keypoints, maxkeypoint);

    struct timeval dst_description_start, dst_description_end;
    gettimeofday( &dst_description_start, NULL );

    //feature description
    Ptr<ORB> dst_extractor = ORB::create(10000, 1.5f, 1, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    dst_extractor->compute(overlap_input, dst_features.keypoints, dst_features.descriptors);
    dst_features.descriptors = dst_features.descriptors.reshape(1, (int)dst_features.keypoints.size());

    gettimeofday( &dst_description_end, NULL );
    //求出两次时间的差值，单位为us
    int dst_descriptiontimeuse = 1000000 * ( dst_description_end.tv_sec - dst_description_start.tv_sec ) + dst_description_end.tv_usec - dst_description_start.tv_usec;
    std::cout << "dst description time is " << dst_descriptiontimeuse << "  us."<< std::endl;
    std::cout << "dst_features descriptors size = " << dst_features.descriptors.size()<< std::endl;

    struct timeval match_start, match_end;
    gettimeofday( &match_start, NULL );
    //build flann index
    Ptr<cv::flann::Index> tree;
    tree.release();
    tree = makePtr<cv::flann::Index>(src_features.descriptors,
            cv::flann::LshIndexParams(5,
                    15, 0),
                    cvflann::FLANN_DIST_HAMMING);

    //flann knn search
    cv::Mat indices, dists;
    Ptr<cv::flann::SearchParams> flann_search_dst = makePtr<flann::SearchParams>(32, 0, false);
    tree->knnSearch(dst_features.descriptors, indices, dists, 2, *flann_search_dst);

    //get match points
    std::vector<Point2f> obj_pt, scene_pt;  //重复区域匹配点list
    obj_pt.clear();
    scene_pt.clear();
    float* dists_ptr;
    int* indeces_ptr;
    for(int i=0;i<dists.rows;i++)
    {
        dists_ptr=dists.ptr<float>(i);
        indeces_ptr = indices.ptr<int>(i);
        if (dists_ptr[0]<(1.f - 0.3)*dists_ptr[1])
        {
            obj_pt.push_back( src_features.keypoints[indeces_ptr[0]].pt );
            scene_pt.push_back( dst_features.keypoints[i].pt );
        }
    }

//    cout << "obj_pt" << endl;
//    cout << obj_pt.size() << endl;
//    cout << scene_pt.size() << endl;

    if (obj_pt.size() < 40)
    {
        dst_extractor.release();
        dst_fastdetector.release();
        flann_search_dst.release();
        std::cout << "obj_pt.size() < 25. " << std::endl;
        Point2d movetemp = move_extern.front();
        for (int i = 0; i < move_extern.size(); ++i) {
            move_extern.push_front(movetemp);
            move_extern.pop_back();
        }
        for(auto move_:move_extern)
            cout << move_;
        cout << endl;
        return false;
    }
    //compute warping matrix
    std::vector<uchar> inliers_mask;
    cv::Mat H = findFundamentalMat(obj_pt, scene_pt, inliers_mask);

    //todo 检测内点数量
    int good_num = 0;
    for (int i=0; i<inliers_mask.size();++i){
        if (inliers_mask[i] != '\0')
            good_num++;
    }

    float conf = good_num /(8 + 0.3 * (obj_pt.size()));
    if (good_num < 30 || conf < 1.0)
//    if (good_num < inliner_num || conf < conf_thresh)
    {
        dst_extractor.release();
        dst_fastdetector.release();
        flann_search_dst.release();
        std::cout << "good_num < inliner_num or conf < conf_thresh" << std::endl;
        Point2d movetemp = move_extern.front();
        for (int i = 0; i < move_extern.size(); ++i) {
            move_extern.push_front(movetemp);
            move_extern.pop_back();
        }
        for(auto move_:move_extern)
            cout << move_;
        cout << endl;
        return false;
    }

    vector<Point2d>keypoint_trans,keypoint_target;
    Mat src_points(1, static_cast<int>(good_num), CV_32FC2);
    Mat dst_points(1, static_cast<int>(good_num), CV_32FC2);

    int inlier_idx = 0;    //表示内点索引
    for (size_t i = 0; i < obj_pt.size(); ++i)    //遍历匹配点对，得到内点
    {
        if (!inliers_mask[i])    //不是内点
            continue;
        Point2f p = obj_pt[i]/overlap_input_scale;    //第一幅图像的内点坐标
        src_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoint_trans.push_back(p);

        p = scene_pt[i]/overlap_input_scale;    //第二幅图像的内点坐标
        dst_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoint_target.push_back(p);
        inlier_idx++;    //索引计数
    }
    for(auto kp:keypoint_target){
        circle(dst,kp,2,Scalar(0,200,0),2);
    }



    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)overlap_input.cols, 0 );
    obj_corners[2] = Point2f( (float)overlap_input.cols, (float)overlap_input.rows );
    obj_corners[3] = Point2f( 0, (float)overlap_input.rows );

    std::vector<Point2f> scene_corners(4);
//
//    //dst points transformation
//    perspectiveTransform(obj_corners, scene_corners, H);
//    //
//    //    if (abs(scene_corners[0].x-scene_corners[1].x) > 4*abs(obj_corners[0].x-obj_corners[1].x) ||
//    //        abs(scene_corners[2].x-scene_corners[3].x) > 4*abs(obj_corners[2].x-obj_corners[3].x) ||
//    //        abs(scene_corners[1].y-scene_corners[2].y) > 4*abs(obj_corners[1].y-obj_corners[2].y) ||
//    //        abs(scene_corners[3].y-scene_corners[0].y) > 4*abs(obj_corners[3].y-obj_corners[0].y) )
//    //
//    //        return false;
//
//    //upsample
//    float scale = 1. / overlap_input_scale;
//    for(int i=0; i<4; i++){
//        dst_pnts[i].x = scene_corners[i].x*scale;
//        dst_pnts[i].y = scene_corners[i].y*scale;
//
//        // 边界保护
//        //if (dst_pnts[i].x < 0) dst_pnts[i].x = 0;
//        //if (dst_pnts[i].y < 0) dst_pnts[i].y = 0;
//        //if (dst_pnts[i].x > dst.cols) dst_pnts[i].x = dst.cols;
//        //if (dst_pnts[i].y > dst.rows) dst_pnts[i].y = dst.rows;
//    }
//    //src points transformation
//    perspectiveTransform( obj_corners, scene_corners, H.inv());
//    //upsample
//    for(int i=0; i<4; i++){
//        src_pnts[i].x = scene_corners[i].x*scale;
//        src_pnts[i].y = scene_corners[i].y*scale;
//
//        // 边界保护
//        // if (src_pnts[i].x < 0) src_pnts[i].x = 0;
//        // if (src_pnts[i].y < 0) src_pnts[i].y = 0;
//        // if (src_pnts[i].x > dst.cols) src_pnts[i].x = dst.cols;
//        // if (src_pnts[i].y > dst.rows) src_pnts[i].y = dst.rows;
//    }
//    gettimeofday( &match_end, NULL );
    //求出两次时间的差值，单位为us
//    int matchtimeuse = 1000000 * ( match_end.tv_sec - match_start.tv_sec ) + match_end.tv_usec - match_start.tv_usec;
//    std::cout << "match time is " << matchtimeuse << "  us."<< std::endl;
//    dst_extractor.release();
//    dst_fastdetector.release();
//    flann_search_dst.release();

//    LOGLN("feature get, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

    Mat img_trans_,img_target_;
    dst.copyTo(img_trans_);
    dst.copyTo(img_target_);

    Mat meanPoint_Mat_tran,meanPoint_Mat_target,std_trans,std_target;

    meanStdDev(keypoint_trans,meanPoint_Mat_tran,std_trans);
    Point2d meanp_trans(meanPoint_Mat_tran.at<double>(0,0),meanPoint_Mat_tran.at<double>(1,0));

//    cout << "overlap_input_scale" << endl;
//    cout << overlap_input_scale << endl;
    for(auto kp:keypoint_target)
        circle(dst,kp,2,Scalar(0,100,0),3);
//    imwrite("imgtrans.jpg",dst);
//

    meanStdDev(dst_points,meanPoint_Mat_target,std_target);
    Point2d meanp_target(meanPoint_Mat_target.at<double>(0,0),meanPoint_Mat_target.at<double>(1,0));

//    for(auto kp:keypoint_target)
//        circle(img_target_,kp,1,Scalar(0,240,0),3);
//    imwrite("imgtarget.jpg",img_target_);

    //todo Calculate R.matrix
    bool Get_Hpic = false;
    Mat img;
    dst.copyTo(img);

    //todo 设置三个旋转角度
    double roll_x, roll_y, roll_z,roll_x_target,roll_y_target,roll_z_target;
    roll_x = Eular_Angle.x;roll_x_target = target_Angle.x;
    roll_y = Eular_Angle.y;roll_y_target = target_Angle.y;
    roll_z = Eular_Angle.z;roll_z_target = target_Angle.z;
//    cout << roll_x<<"\t" << roll_y << roll_z << endl;
    double pitch,roll,yaw;
    //todo target pic
    pitch = M_PI_2* (roll_x-roll_x_target)/90;//绕x
    roll  = M_PI_2* (roll_y-roll_y_target)/90;//绕y
    yaw   = M_PI_2* (roll_z-roll_z_target)/90;//绕z
    //防止越界
    if(pitch >= M_PI)       pitch -= M_PI;
    else if(pitch <= M_PI)  pitch += M_PI;
    if(roll >= M_PI)        roll -= M_PI;
    else if(roll <= M_PI)   roll += M_PI;
    if(yaw >= M_PI)         yaw -= M_PI;
    else if(yaw <= M_PI)    yaw += M_PI;


    double Cx,Cy;
    Cx = img.cols/2;
    Cy = img.rows/2;
    double fx = 800;
    double fy = 800;////
    double Deepth = 30 * fx;

    //todo 从旋转角度得到旋转矩阵

    cv::Vec3d theta(pitch, roll, yaw);
    cv::Mat R_x = (cv::Mat_<double>(3, 3) <<
                                          1, 0, 0, 0, cos(theta[0]), -sin(theta[0]), 0, sin(theta[0]), cos(theta[0]));
    cv::Mat R_y = (cv::Mat_<double>(3, 3) <<
                                          cos(theta[1]), 0, sin(theta[1]), 0, 1, 0, -sin(theta[1]), 0, cos(theta[1]));
    cv::Mat R_z = (cv::Mat_<double>(3, 3) <<
                                          cos(theta[2]), -sin(theta[2]), 0, sin(theta[2]), cos(theta[2]), 0, 0, 0, 1);
    cv::Mat R = R_z * R_y * R_x;

    Mat Point_in3D = (Mat_<double>(3,1) << 1, 0, 0);
    cv::Mat Point_projected = R * Point_in3D;

    //todo 得到去中心的相机内参2
    Mat K_f =(cv::Mat_<double>(3, 3) << fx,0,0,0,fy,0,0,0,1);
    //    cout << "K" << K << endl;
    Mat K_inv;
    cv::invert(K_f,K_inv);
    //simulate the size of the pic
    vector <Mat> corners (4);
    corners[0] =(Mat_<double>(3,1) <<  Cx,Cy,1);
    corners[1] =(Mat_<double>(3,1) << -Cx,Cy,1);
    corners[2] =(Mat_<double>(3,1) << -Cx,-Cy,1);
    corners[3] =(Mat_<double>(3,1) <<  Cx,-Cy,1);
    vector<Mat> corners_trans(4);
    vector< Point2d> roi_corners;
    vector< Point2d> dst_corners;

    for (int j = 0; j < 4; ++j) {
        const double* Z = reinterpret_cast<const double*>(corners[j].data);
        roi_corners.push_back(Point2f( (double)(Z[0] / 1.), (double)(Z[1] / 1.) ));
        corners_trans[j] = (Deepth*K_inv.t()*corners[j]);
    }

    //todo 得到初始平面点和投影平面点
    for (int j = 0; j < 4; ++j) {
        corners_trans[j] = R.t()*corners_trans[j];
        const double* h = reinterpret_cast<const double*>(corners_trans[j].data);
        corners_trans[j] = (K_f * corners_trans[j]/h[2]);
        h = reinterpret_cast<const double*>(corners_trans[j].data);
        dst_corners.push_back(Point2f( (double)(h[0] / 1.), (double)(h[1] / 1.) ));
    }
    //todo findhomograph
    // 现在的H矩阵是在中心为0的基础上投影变换的
    H = findHomography(roi_corners,dst_corners);

    //todo Get_Used_inv_H * function
    vector< Point2d> Cpoint_output;
    vector <Mat> corners_t(4);
    vector<Point2d> Cpoint_input;
    for (int i = 0; i < 4; ++i) {
        corners_t[i] =  H*corners[i];
        const double* h = reinterpret_cast<const double*>(corners_t[i].data);
        // cout << "corners_t " <<  corners_t[i] << endl;
        Cpoint_output.push_back(Point2f( (double)(h[0] / h[2]), (double)(h[1] / h[2]) ));
        const double* h2 = reinterpret_cast<const double*>(corners[i].data);
        Cpoint_input.push_back(Point2f( (double)(h2[0] + Cx), (double)(h2[1] + Cy)));
    }

    //todo 找到角坐标
    Point2d Zero_bias(0,0);
    for (int i = 0; i < 4; ++i) {
        if(Zero_bias.x > Cpoint_output[i].x)
            Zero_bias.x = Cpoint_output[i].x;
        if(Zero_bias.y > Cpoint_output[i].y)
            Zero_bias.y = Cpoint_output[i].y;
    }
    for (int i = 0; i < 4; ++i) {
        Cpoint_output[i].x -= Zero_bias.x;
        Cpoint_output[i].y -= Zero_bias.y;
    }

    Mat H_pic = findHomography(Cpoint_input,Cpoint_output);

    Mat Trans_after;
    perspectiveTransform(src_points,Trans_after,H_pic);
    // cout<< Trans_after.size << endl;

    Mat meanfeature_trans =(cv::Mat_<double>(3, 1) << meanp_trans.x ,meanp_trans.y ,1);
    meanfeature_trans = H_pic*meanfeature_trans;
    Point2d fordstPic(meanfeature_trans.at<double>(0,0)/meanfeature_trans.at<double>(2,0),meanfeature_trans.at<double>(1,0)/meanfeature_trans.at<double>(2,0));
    // Point2d
    Mat move_mat = Trans_after - dst_points;

    // Point2d move_scale(meanfeature_trans.at<double>(0,0),meanfeature_trans.at<double>(1,0));

    Mat t1,t2,std_t1;
    meanStdDev(Trans_after,t2,std_t1);
    src_pnts[0].x = t2.at<double>(0,0);
    src_pnts[0].y = t2.at<double>(1,0);

    meanStdDev(dst_points,t2,std_t1);
    src_pnts[1].x = t2.at<double>(0,0);
    src_pnts[1].y = t2.at<double>(1,0);

    meanStdDev(move_mat,t1,std_t1);
    // cout << t1.size << endl;
    Point2d move_scale(t1.at<double>(0,0),t1.at<double>(1,0));

    move_extern.push_front(move_scale);
    move_extern.pop_back();

    Point2d Summove(0,0);
    for(auto move:move_extern){
        Summove += move;}
    Summove.x = Summove.x/(move_extern.size());
    Summove.y = Summove.y/(move_extern.size());

    vector<double> x_,y_;
    int move_idx =0;
    for(Point2d move_:move_extern){
        x_.push_back(move_.x);
        y_.push_back(move_.y);
    }

    double sum = std::accumulate(std::begin(x_), std::end(x_), 0.0);
    double mean =  sum / x_.size(); //均值

    double accum  = 0.0;
    std::for_each (std::begin(x_), std::end(x_), [&](const double d) {
        accum  += (d-mean)*(d-mean);
    });
    double stdev = sqrt(accum/(x_.size()-1)); //方差
    cout << "stdev" << endl;
    cout << stdev << endl;

//    cout << Summove << endl;
    //3.update measurement
    //    cout << predict_pt<< endl;
    src_pnts[2] = Summove;
//    float scale = 1. / overlap_input_scale;

    for (int i = 0; i < 4; ++i) {
        Cpoint_output[i] -= Summove;
    }

    for(int i=0; i<4; i++){
        dst_pnts[i].x = Cpoint_output[i].x;
        dst_pnts[i].y = Cpoint_output[i].y;
    }

    return true;



}
//
// Created by aruchid on 2019/12/19.
//

