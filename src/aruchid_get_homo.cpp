//
// Created by aruchid on 2019/12/8.
//

#include "../include/aruchid_get_homo.h"


Mat Simplely_findHomo(vector<KeyPoint> keypoints_trans,vector<KeyPoint> keypoints_target){

    Mat src_points(1, static_cast<int>(keypoints_trans.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(keypoints_target.size()), CV_32FC2);
    int inlier_idx = 0;    //表示内点索引
    for (size_t i = 0; i < keypoints_trans.size(); ++i)    //遍历匹配点对，得到内点
    {

//            Point2f p = keypoints1_SIFT[m.queryIdx].pt;    //第一幅图像的内点坐标
        src_points.at<Point2f>(0, inlier_idx) = keypoints_trans[i].pt;    //赋值
//            keypoints_1->push_back(keypoints1_SIFT[m.queryIdx]);

//            p = keypoints2_SIFT[m.trainIdx].pt;    //第二幅图像的内点坐标
        dst_points.at<Point2f>(0, inlier_idx) = keypoints_target[i].pt;    //赋值
//            keypoints_2->push_back(keypoints2_SIFT[m.trainIdx]);
        inlier_idx++;    //索引计数
//        matches_output->push_back(matches[i]);
    }
    std::vector<uchar> inlier_mask;
    Mat H_temp = findHomography(src_points,dst_points,RANSAC,4,inlier_mask,500,0.9999);
    return H_temp;
}

Mat ECC_refineHomo(const Mat imgTemplate,
        const Mat imgTrans,
        vector<KeyPoint> keypoints_trans,
        vector<KeyPoint> keypoints_target){

    Mat imgTemp_,imgTrans_;
    imgTemplate.copyTo(imgTemp_);
    cvtColor(imgTemp_,imgTemp_,COLOR_RGB2GRAY);
    imgTrans.copyTo(imgTrans_);
    cvtColor(imgTrans_,imgTrans_,COLOR_RGB2GRAY);


    Mat src_points(1, static_cast<int>(keypoints_trans.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(keypoints_target.size()), CV_32FC2);
    int inlier_idx = 0;    //表示内点索引
    for (size_t i = 0; i < keypoints_trans.size(); ++i)    //遍历匹配点对，得到内点
    {

//            Point2f p = keypoints1_SIFT[m.queryIdx].pt;    //第一幅图像的内点坐标
        src_points.at<Point2f>(0, inlier_idx) = keypoints_trans[i].pt;    //赋值
//            keypoints_1->push_back(keypoints1_SIFT[m.queryIdx]);

//            p = keypoints2_SIFT[m.trainIdx].pt;    //第二幅图像的内点坐标
        dst_points.at<Point2f>(0, inlier_idx) = keypoints_target[i].pt;    //赋值
//            keypoints_2->push_back(keypoints2_SIFT[m.trainIdx]);
        inlier_idx++;    //索引计数
//        matches_output->push_back(matches[i]);
    }
    std::vector<uchar> inlier_mask;
    Mat H_temp = findHomography(src_points,dst_points,RANSAC,4,inlier_mask,500,0.9999);


    int warp_mode = MOTION_HOMOGRAPHY;
    //todo start ECC
    //最大迭代数
    int number_of_iterations = 100;
    //迭代精度
    double termination_eps = 1e-10;
    //迭代标准
    H_temp.convertTo(H_temp,CV_32F);
//    Mat warp_matrix = Mat::eye(3, 3, CV_32F);
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS,
                          number_of_iterations, termination_eps);
    //计算变换矩阵
    cout << "transECC"<< endl;
    findTransformECC(imgTemp_,imgTrans_,H_temp,warp_mode,criteria);
    //计算对齐后的图像
    cout << "transECC Succcc"
            ""<< endl;
    cout << H_temp << endl;

//    cout << warp_matrix << endl;
//    warp_matrix.convertTo(warp_matrix,CV_32F);
//    H_temp = Mat<double>warp_matrix

    H_temp.convertTo(H_temp,CV_64FC1);
    return H_temp;
}
