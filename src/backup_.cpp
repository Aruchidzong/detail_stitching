//
// Created by aruchid on 2019/12/21.
//


#include "../include/overlap.hpp"
#include "../stitch_connector.h"
#include "../include/aruchid_pipeline.h"
#include "../include/aruchid_featuresfind.h"
//#include "../stitch_connector.h"
#include "../include/aruchid_rotation.h"
#include "iostream"
#include <string>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>


//RNG rng;
//1.kalman filter setup

list<float> confidence_List(10);
vector<Point> corner_extern(4);
list<Point2d> move_extern(3);

bool flat_false =false;


bool set_src_feature(cv::Mat &src, vector<KeyPoint> &src_keypoints,Mat& src_descriptor){

    const int winHeight=src.rows;
    const int winWidth=src.cols;

    if (src.empty())
    {
        std::cout << "src is empty." << std::endl;
        return false;
    }
    double overlap_input_megapix = 0.5;  //输入图片默认尺寸
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
    resize(src, overlap_input, Size(),
           overlap_input_scale, overlap_input_scale,
           INTER_NEAREST);

    Ptr<FeatureDetector> src_detector = AKAZE::create();
    //    Ptr<FeatureDetector> src_fastdetector = ORB::create(5000,5);
    src_detector->detect(overlap_input, src_keypoints);
    cout << "src_keypoints.size()" << endl;
    cout << src_keypoints.size() << endl;

    if (src_keypoints.size() < 100)
    {
        src_detector.release();
        return false;
    }

//    KeyPointsFilter::runByImageBorder(src_keypoints, overlap_input.size(), 31);
//    KeyPointsFilter::retainBest(src_keypoints, maxkeypoint);

    //feature description
    Ptr<DescriptorExtractor> src_extractor = AKAZE::create();
    src_extractor->compute(overlap_input,src_keypoints, src_descriptor);
//    src_descriptor = src_descriptor.reshape(1, (int)src_keypoints.size());

    cout << src_keypoints.size() << endl;
    cout << src_descriptor.size << endl;

    src_extractor.release();
    src_detector.release();
    return true;
}



bool overlap_point(  cv::Mat &dst,
                     vector<KeyPoint> &src_keypoints,
                     Mat src_descriptor,
                     const Point3d target_Angle,
                     const Point3d Eular_Angle,
                     vector<Point> &src_pnts,
                     vector<Point> &dst_pnts){

    int area_color;
    double app_start_time = getTickCount();
    if (dst.empty())
    {
        std::cout << "dst is empty. " << std::endl;
        //        flat_false = true;
        return false;
    }

    if(src_keypoints.size() == 0)
    {
        std::cout << "src_features kpt size = 0." << std::endl;
        //        flat_false = true;
        return false;
    }

    double overlap_input_megapix = 0.5;  //输入图片默认尺寸
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

    //    overlap_input_scale = 1;
    // }
    if(dst.channels() == 3)
        cv::cvtColor(dst, overlap_input, COLOR_BGR2GRAY);
    else
        dst.copyTo(overlap_input);
    resize(overlap_input, overlap_input, Size(), overlap_input_scale, overlap_input_scale, INTER_NEAREST);

    //feature detect
    //    ImageFeatures dst_features;
    //    Ptr< FastFeatureDetector> dst_fastdetector = FastFeatureDetector::create (20, true, FastFeatureDetector::TYPE_9_16);

    vector<KeyPoint> dst_keypoints;
    Mat dst_descriptors;
    Ptr<FeatureDetector> dst_detector = AKAZE::create();

    //todo 这里是重叠区域
    //    Mat Overlap_Area ;
    //    vector<vector<Point> > vpts;
    //    vpts.push_back(corner_extern);
    //    overlap_input.copyTo(Overlap_Area);
    //    Overlap_Area = Scalar::all(0);


    //dst_detector
    dst_detector->detect(overlap_input, dst_keypoints);


    LOGLN("dst_detectors , total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    app_start_time = getTickCount();
    //todo feature descriptions
    Ptr<DescriptorExtractor> dst_extractor = AKAZE::create();
    dst_extractor->compute(overlap_input, dst_keypoints, dst_descriptors);


    LOGLN("dst_extractor , total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    app_start_time = getTickCount();
    //todo-- 第四步:匹配点对筛选RANSAC
    // Check if it makes sense to find homography
    vector<DMatch> matches;
    if(dst_descriptors.empty())
        return false;
    float match_conf_ = 0.3f;
    KnnMatcher(src_descriptor, dst_descriptors, matches, match_conf_);
    //判断两幅图像的匹配点对的数量是否达到了设置的阈值，如果小于该阈值，说明两幅图像没有重叠的地方，无需再进行拼接
//    cout << "img_size" << img_size << endl;
    // Construct point-point correspondences for homography estimation
    //定义两个矩阵，用于保存两幅图像的匹配点对的坐标
    std::vector<Point2f> src_pt, dst_pt;  //重复区域匹配点list
    Mat src_points(1, static_cast<int>(matches.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(matches.size()), CV_32FC2);
    vector<Point2d>keypoint_trans,keypoint_target;
    //遍历所有匹配点对，得到匹配点对的特征点坐标
    for (size_t i = 0; i < matches.size(); ++i) {
        const DMatch &m = matches[i];
        //得到第一幅图像的当前匹配点对的特征点坐标
        Point2f p = src_keypoints[m.queryIdx].pt;
        //以图像的中心处为坐标原点，得到此时的特征点坐标，因为默认情况下是以图像的左上角为坐标原点的
//        src_points.at<Point2f>(0, static_cast<int>(i)) = p;    //特征点坐标赋值
        src_pt.push_back(p);
        //得到第二幅图像的当前匹配点对的特征点坐标
        p = dst_keypoints[m.trainIdx].pt;
        //以图像的中心处为坐标原点，得到此时的特征点坐标，因为默认情况下是以图像的左上角为坐标原点的
//        dst_points.at<Point2f>(0, static_cast<int>(i)) = p;    //特征点坐标赋值
        dst_pt.push_back(p);
    }
    if (dst_pt.size() < 50)
    {
        dst_detector.release();
        return false;
    }
//    cout << src_pt.size() << endl;
//    cout << dst_pt.size() << endl;
//    cout << "featuresfind.dst_points " << dst_points.size << endl;
    std::vector<uchar> inlier_mask;
    // TODO Find pair-wise motion
    //利用所有的匹配点对得到单应矩阵，findHomography函数在后面有详细的讲解，src_points和dst_points分别表示两幅图像的特征点，
    // 它们是匹配点对的关系，matches_info.inliers_mask表示内点的掩码，即哪些特征点属于内点，CV_RANSAC表示使用RANSAC的方法来得到单应矩阵
//    Mat dst_ransac;
    Mat H_return;
    findFundamentalMat(src_pt, dst_pt, inlier_mask, FM_RANSAC);
//    findHomography(dst_points, src_points, inlier_mask, RANSAC);
//    H_return = findHomography(src_points, dst_points, RANSAC, 4.0, inlier_mask, 500, 0.9999);

    // todo from here: rotation overlap area
    float confidence;
    int num_inliers = 0;    //匹配点对的内点数先清零
    //由内点掩码得到内点数
    for (size_t i = 0; i < inlier_mask.size(); ++i)
        if (inlier_mask[i]){
            num_inliers++;}
    if (num_inliers < 25)
        return false;
//    findHomography(scene_pt, obj_pt, RANSAC, 4.0, inlier_mask, 500, 0.9999);
    // These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
    // using Invariant Features"
    //计算匹配置信度c，式23
    confidence = num_inliers / (8 + 0.3 * src_pt.size());
    // Set zero confidence to remove matches between too close images, as they don't provide
    // additional information anyway. The threshold was set experimentally.
    //如果匹配置信度太大（大于3，3为实验数据），则认为这两幅图像十分接近，可以被看成是一幅图像，因此无需匹配，并要把置信度重新赋值为0
    confidence = confidence > 4. ? 0. : confidence;
    std::cout << num_inliers << " "<<confidence << std::endl;


    LOGLN("confidence , total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    if(confidence < 1)
        return false;

//        cout << obj_pt.size()<< endl;
    //todo delete after publish

    int inlier_idx = 0;    //表示内点索引
    for (size_t i = 0; i < src_pt.size(); ++i)    //遍历匹配点对，得到内点
    {
        if (!inlier_mask[i])    //不是内点
            continue;
        Point2f p = src_pt[i]/overlap_input_scale;    //第一幅图像的内点坐标
        src_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoint_trans.push_back(p);
        p = dst_pt[i]/overlap_input_scale;    //第二幅图像的内点坐标
        dst_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoint_target.push_back(p);
        inlier_idx++;    //索引计数
    }

//    LOGLN("feature maches, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

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
    vector< Point2f> roi_corners;
    vector< Point2f> dst_corners;

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
    Mat H = findHomography(roi_corners,dst_corners);

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

    move_extern.push_back(move_scale);
    move_extern.pop_front();

    Point2d Summove(0,0);
    for(auto move:move_extern){
        Summove += move;}
    Summove.x = Summove.x/(move_extern.size());
    Summove.y = Summove.y/(move_extern.size());
//    cout << Summove << endl;
    //3.update measurement
    //    cout << predict_pt<< endl;

    float scale = 1. / overlap_input_scale;

    for (int i = 0; i < 4; ++i) {
        Cpoint_output[i] -= Summove;
    }

    for(int i=0; i<4; i++){
        dst_pnts[i].x = Cpoint_output[i].x;
        dst_pnts[i].y = Cpoint_output[i].y;
    }

    corner_extern.assign(dst_pnts.begin(),dst_pnts.end());
    confidence_List.push_back(confidence);
    confidence_List.pop_front();




//    float sum_conf = 0;
//    for(auto move:move_extern){
////        sum_conf += move;
//        cout << move;}
//    cout<< endl;
//    cout << src_pnts[0]<< endl;

    float sum_conf = 0;
    for(auto confidence:confidence_List)
        sum_conf += confidence;
    float filter_conf = sum_conf/confidence_List.size();
//        cout << filter_conf<< endl;
//    cout << endl;
    if(flat_false)
        return false;
    return true;
}


