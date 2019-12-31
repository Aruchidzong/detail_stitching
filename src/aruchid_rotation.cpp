//
// Created by aruchid on 2019/11/30.
//
//#include <direct.h>
#include <iostream>
#include "../include/aruchid_rotation.h"
#include <sys/stat.h>
#include <sys/types.h>
#include "../include/aruchid_featuresfind.h"
void Find_Overlap_area(Mat img_trans,
                       Mat img_target,
                       const Point3d Eular_Angle,
                       const Point3d target_Angle,
                       vector<Point> &src_pnts,
                       vector<Point> &dst_pnts
                       )
{


    double app_start_time = getTickCount();
    vector<KeyPoint> keypoint_trans,keypoint_target;
    Mat img_trans_,img_target_;
    img_trans.copyTo(img_trans_);
    img_target.copyTo(img_target_);

//todo rotated finished

    Mat wrong;
    float match_conf_ = 0.25f;
    int num_matches_thresh1_ = 6;
    int num_matches_thresh2_ = 6;
    keypoint_trans.clear();
    keypoint_target.clear();

    //todo 这里开始提取各类匹配点
    //-- 初始化
    Mat descriptors1_SIFT, descriptors2_SIFT;
    vector<KeyPoint> keypoints1_SIFT, keypoints2_SIFT;

    //todo use SIFT
    Ptr<FeatureDetector> detectorSIFT = ORB::create();
    Ptr<DescriptorExtractor> descriptorSIFT = ORB::create();
//    Ptr<FeatureDetector> detectorSIFT = AKAZE::create();
//    Ptr<DescriptorExtractor> descriptorSIFT = AKAZE::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    // Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce" );

    //-- 第一步:检测 Oriented FAST 角点位置
    detectorSIFT->detect ( img_trans_,keypoints1_SIFT );
    detectorSIFT->detect ( img_target_,keypoints2_SIFT );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptorSIFT->compute ( img_trans_, keypoints1_SIFT, descriptors1_SIFT );
    descriptorSIFT->compute ( img_target_, keypoints2_SIFT, descriptors2_SIFT );


    LOGLN("feature get, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

    app_start_time = getTickCount();
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用KNN
    vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
//    matcher->match ( descriptors_1, descriptors_2, match );
    KnnMatcher(descriptors1_SIFT, descriptors2_SIFT, matches, match_conf_);
//    vector<Mat> imgdraw;
//    imgdraw.push_back(img1);
//    imgdraw.push_back(img2);

//    Mat dstKNN;
//    drawMatches(img1,keypoints1_SIFT,img2,keypoints2_SIFT,matches,dstKNN);
//    imwrite("./result/dstKNN.jpg",dstKNN);
//    matches = match;
//    cout << "//第四步:匹配点对筛选RANSAC"<< endl;

    //todo-- 第四步:匹配点对筛选RANSAC
    // Check if it makes sense to find homography
    //判断两幅图像的匹配点对的数量是否达到了设置的阈值，如果小于该阈值，说明两幅图像没有重叠的地方，无需再进行拼接
    if (matches.size() < static_cast<size_t>(num_matches_thresh1_))
    //        return wrong;

    Size img_size = Size(img_trans_.cols,img_target_.rows);
    //    cout << "img_size" << img_size << endl;
    // Construct point-point correspondences for homography estimation
    //定义两个矩阵，用于保存两幅图像的匹配点对的坐标
    Mat src_points(1, static_cast<int>(matches.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(matches.size()), CV_32FC2);
    //遍历所有匹配点对，得到匹配点对的特征点坐标
    for (size_t i = 0; i < matches.size(); ++i) {
        const DMatch &m = matches[i];
        //得到第一幅图像的当前匹配点对的特征点坐标
        Point2f p = keypoints1_SIFT[m.queryIdx].pt;
        //以图像的中心处为坐标原点，得到此时的特征点坐标，因为默认情况下是以图像的左上角为坐标原点的
    //        p.x -= img_size.width * 0.5f;
    //        p.y -= img_size.height * 0.5f;
        src_points.at<Point2f>(0, static_cast<int>(i)) = p;    //特征点坐标赋值

        //得到第二幅图像的当前匹配点对的特征点坐标
        p = keypoints2_SIFT[m.trainIdx].pt;
        //以图像的中心处为坐标原点，得到此时的特征点坐标，因为默认情况下是以图像的左上角为坐标原点的
    //        p.x -= img_size.width * 0.5f;
    //        p.y -= img_size.height * 0.5f;
        dst_points.at<Point2f>(0, static_cast<int>(i)) = p;    //特征点坐标赋值
    }

    //    cout << "featuresfind.dst_points " << dst_points.size << endl;
    std::vector<uchar> inlier_mask;
    // TODO Find pair-wise motion
    //利用所有的匹配点对得到单应矩阵，findHomography函数在后面有详细的讲解，src_points和dst_points分别表示两幅图像的特征点，
    // 它们是匹配点对的关系，matches_info.inliers_mask表示内点的掩码，即哪些特征点属于内点，CV_RANSAC表示使用RANSAC的方法来得到单应矩阵
    Mat dst_ransac;
    Mat H_return;
//    findFundamentalMat(dst_points, src_points, inlier_mask, FM_RANSAC);
//    findHomography(dst_points, src_points, inlier_mask, RANSAC);
//    estimateAffine2D(src_points,dst_points,inlier_mask,RANSAC);
    H_return = findHomography(src_points, dst_points, RANSAC, 4.0, inlier_mask, 500, 0.9999);

//    Draw_match(keypoints1_SIFT,keypoints2_SIFT,matches,imgdraw,inlier_mask);
    int num_inliers = 0;    //匹配点对的内点数先清零
    //由内点掩码得到内点数
    for (size_t i = 0; i < inlier_mask.size(); ++i)
        if (inlier_mask[i]){
            num_inliers++;}
//    cout << "num_inliers = " <<num_inliers <<endl;
    int inlier_idx = 0;    //表示内点索引
    for (size_t i = 0; i < matches.size(); ++i)    //遍历匹配点对，得到内点
    {
        if (!inlier_mask[i])    //不是内点
            continue;
        const DMatch &m = matches[i];    //赋值

        Point2f p = keypoints1_SIFT[m.queryIdx].pt;    //第一幅图像的内点坐标
        src_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoint_trans.push_back(keypoints1_SIFT[m.queryIdx]);

        p = keypoints2_SIFT[m.trainIdx].pt;    //第二幅图像的内点坐标
        dst_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoint_target.push_back(keypoints2_SIFT[m.trainIdx]);
        inlier_idx++;    //索引计数
//        matches_output->push_back(matches[i]);
    }
    LOGLN("feature maches, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

    app_start_time = getTickCount();

    Mat meanPoint_Mat_tran,meanPoint_Mat_target,std_trans,std_target;
    meanStdDev(src_points,meanPoint_Mat_tran,std_trans);
//    cout << "meanp_trans = " << meanPoint_Mat_tran<< endl;
//    cout << "std = " << std_trans << endl;
    Point2d meanp_trans(meanPoint_Mat_tran.at<double>(0,0),meanPoint_Mat_tran.at<double>(1,0));

//    circle(img_trans_,meanp_trans,10,Scalar(0,240,0),10);
//    imwrite("imgtrans.jpg",img_trans_);

    meanStdDev(dst_points,meanPoint_Mat_target,std_target);
//    cout << "meanp_target = " << meanPoint_Mat_target<< endl;
//    cout << "std = " << std_target << endl;
    Point2d meanp_target(meanPoint_Mat_target.at<double>(0,0),meanPoint_Mat_target.at<double>(1,0));

//    circle(img_target_,meanp_target,10,Scalar(0,240,0),10);
//    imwrite("imgtarget.jpg",img_target_);
//    cout << keypoint_trans.size() << endl;


//    cout << "meanp_trans-meanp_target "<< endl;
//    Point Trans = meanp_trans-meanp_target;
//    cout << Trans << endl;
//    cout << meanp_trans-meanp_target << endl;
//    cout << std_target/std_trans << endl;
    Mat Zooming = std_target/std_trans;



    //todo pic log
//    for (int k = 0; k < keypoint_trans.size(); ++k)
//    {
//        circle(trans_t,keypoint_trans[k].pt,10,Scalar(207,0,112),2);
//        circle(img_target_,keypoint_target[k].pt,10,Scalar(207,0,112),2);
//    }
//    imwrite("trans_t.jpg",trans_t);
//    imwrite("target_t.jpg",target_t);
//    drawMatches(img_trans,keypoint_trans,img_target,keypoint_target)
    //todo Calculate R.matrix
    bool Get_Hpic = false;
    Mat img;
    img_target.copyTo(img);
    //    cout << "img size" << img.size << endl;
    //    imwrite("img.jpg",img);
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
    double Deepth = 30 * fx;//52mm

    //todo 从旋转角度得到旋转矩阵
//    Eigen::Matrix3d rotation;
    cv::Vec3d theta(pitch, roll, yaw);
    cv::Mat R_x = (cv::Mat_<double>(3, 3) <<
                                          1, 0, 0, 0, cos(theta[0]), -sin(theta[0]), 0, sin(theta[0]), cos(theta[0]));
    cv::Mat R_y = (cv::Mat_<double>(3, 3) <<
                                          cos(theta[1]), 0, sin(theta[1]), 0, 1, 0, -sin(theta[1]), 0, cos(theta[1]));
    cv::Mat R_z = (cv::Mat_<double>(3, 3) <<
                                          cos(theta[2]), -sin(theta[2]), 0, sin(theta[2]), cos(theta[2]), 0, 0, 0, 1);
    cv::Mat R = R_z * R_y * R_x;

    Mat Point_in3D = (Mat_<double>(3,1) << 1, 0, 0);
    //投影边角在新坐标系下的3D坐标
    cv::Mat Point_projected = R * Point_in3D;
    //cout << Point_projected << endl;

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

    Mat write_pic;
    img.copyTo(write_pic);
    vector< Point2d> Cpoint_output;

    Mat mean_trans_centre  = (Mat_<double>(3,1) << meanp_trans.x-Cx,  meanp_trans.y-Cy,  1);
    Mat mean_target_centre = (Mat_<double>(3,1) << meanp_target.x-Cx, meanp_target.y-Cy, 1);

//    cout << "mean_trans_centre" << endl;
////    cout << mean_trans_centre << endl;
//    cout << mean_trans_centre << endl;
//    cout << mean_target_centre << endl;
//    Mat mean_trans_centre(meanp_trans.x,meanp_trans.y);
//    Mat mean_target_centre();
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
    //    for (int i = 0; i < 4; ++i) {
    //        cout << "Cpoint_input " <<  Cpoint_input[i] << endl;
    //    }
    //todo 找到角坐标
    Point2d Zero_bias(0,0);
    Point2d OutputBorder(0,0);
    for (int i = 0; i < 4; ++i) {
        //        cout << "Cpoint_output " <<  Cpoint_output[i] << endl;
        if(Zero_bias.x > Cpoint_output[i].x)
            Zero_bias.x = Cpoint_output[i].x;
        if(Zero_bias.y > Cpoint_output[i].y)
            Zero_bias.y = Cpoint_output[i].y;
    }
    //    cout << Zero_bias << endl;
    for (int i = 0; i < 4; ++i) {
        Cpoint_output[i].x -= Zero_bias.x;
        Cpoint_output[i].y -= Zero_bias.y;

//        cout << "Cpoint_output " <<  Cpoint_output[i] << endl;
        if(OutputBorder.x < Cpoint_output[i].x)
            OutputBorder.x = Cpoint_output[i].x;
        if(OutputBorder.y < Cpoint_output[i].y)
            OutputBorder.y = Cpoint_output[i].y;
    }
//    cout << "::::::" << endl;
//    cout << Cpoint_input << endl;
//    cout << Cpoint_output << endl;

    LOGLN("rotate, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

    Mat meanfeature_trans =(cv::Mat_<double>(3, 1) << meanp_trans.x ,meanp_trans.y ,1);
//    cout << "meanfeature_trans" <<endl;
//    cout << meanfeature_trans <<endl;

    Mat H_pic = findHomography(Cpoint_input,Cpoint_output);
//    Mat meanoutput,meanoutputhg;
//    cout << meanPoint_Mat_target << endl;
//    convertPointsToHomogeneous(meanPoint_Mat_target.t(),meanoutput);
//    cout << "meanoutput" << endl;
//    transpose(meanoutput,meanoutputhg);
//    cout << meanoutputhg << endl;
//    perspectiveTransform(meanfeature_trans,meanfeature_trans,H_pic);
    meanfeature_trans = H_pic*meanfeature_trans;

    Point2d fordstPic(meanfeature_trans.at<double>(0,0)/meanfeature_trans.at<double>(2,0),meanfeature_trans.at<double>(1,0)/meanfeature_trans.at<double>(2,0));
//    Point fordstPic()
//    cout << fordstPic << endl;
//    cout << meanp_target << endl;
//    cout << fordstPic - meanp_target << endl;
    Point2d move_scale (fordstPic.x - meanp_target.x,fordstPic.y - meanp_target.y) ;
//    cout << move_scale << endl;
//    Point2d meanp_target2(meanPoint_Mat_target.at<double>(0,0),meanPoint_Mat_target.at<double>(1,0));
    for (int i = 0; i < 4; ++i) {
        Cpoint_output[i] -= move_scale;
    }
//    Point
    Mat dst_pic;
//    warpPerspective(img_trans_,dst_pic,H_pic,Size());
//    circle(dst_pic,fordstPic,10,Scalar(0,240,240),10);
//    imwrite("warp.jpg",dst_pic);
//    cout << ":::perspectiveTransform:::" << endl;

//    cout << Cpoint_input << endl;
//    cout << Cpoint_output << endl;
//    Mat tempmean;
//    convertPointsToHomogeneous(meanPoint_Mat_tran.t(),tempmean);

//    cout << meanPoint_Mat_target << endl;
//    Point Trans = mean_target_centre - mean_trans_centre;
//    for (int i = 0; i < 4; ++i) {
//        Cpoint_output[i].x -= tempmean.at<double>(0,0) - meanPoint_Mat_target.at<double>(0,0) ;
//        Cpoint_output[i].y -= tempmean.at<double>(1,0) - meanPoint_Mat_target.at<double>(1,0) ;
//    }
//    Point trans_mean ;
//    trans_mean.x = mean_trans_centre.at<double>(0,0);
//    trans_mean.y = mean_trans_centre.at<double>(1,0);
//    circle(img,trans_mean,10,Scalar(0,240,0),10);
    //todo 自定义perspective
//    Mat dst;
//    H= Center_WarpPerspective(H,img,dst,corners,1,"downloadpath");
    line(img,Cpoint_output[0],Cpoint_output[1],Scalar(0,240,0),5);
    line(img,Cpoint_output[1],Cpoint_output[2],Scalar(0,240,0),5);
    line(img,Cpoint_output[2],Cpoint_output[3],Scalar(0,240,0),5);
    line(img,Cpoint_output[3],Cpoint_output[0],Scalar(0,240,0),5);
    imwrite("overlap_area.jpg",img);
//    return Mat();
}

Mat  Rotation::Get_RotationHomo(   Mat img_input,
                        const Point3d Eular_Angle,
                        const Point3d target_Angle,
                        CameraParams Kamera
                        )
{
    bool Get_Hpic = false;
    Mat img;
    img_input.copyTo(img);
    //    cout << "img size" << img.size << endl;
    //    imwrite("img.jpg",img);
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
    //cout << pitch<<"\t" << roll << "\t" << yaw << endl;
//    double Cx,Cy;
//    double fx = Kamera.focal;
//    double fy = Kamera.focal;
//    Cx = Kamera.ppx;
//    Cy = Kamera.ppy;

    double Cx,Cy;
    Cx = img_input.cols/2;
    Cy = img_input.rows/2;
    double fx = 800;
    double fy = 800;////
    double Deepth = 30 * fx;//52mm

    //todo 从旋转角度得到旋转矩阵
//    Eigen::Matrix3d rotation;
    cv::Vec3d theta(pitch, roll, yaw);
    cv::Mat R_x = (cv::Mat_<double>(3, 3) <<
            1, 0, 0, 0, cos(theta[0]), -sin(theta[0]), 0, sin(theta[0]), cos(theta[0]));
    cv::Mat R_y = (cv::Mat_<double>(3, 3) <<
            cos(theta[1]), 0, sin(theta[1]), 0, 1, 0, -sin(theta[1]), 0, cos(theta[1]));
    cv::Mat R_z = (cv::Mat_<double>(3, 3) <<
            cos(theta[2]), -sin(theta[2]), 0, sin(theta[2]), cos(theta[2]), 0, 0, 0, 1);
    cv::Mat R = R_z * R_y * R_x;

    Mat Point_in3D = (Mat_<double>(3,1) << 1, 0, 0);
    //投影边角在新坐标系下的3D坐标
    cv::Mat Point_projected = R * Point_in3D;
    //cout << Point_projected << endl;

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
//        h = reinterpret_cast<const double*>(corners_trans[j].data);
    }

    //todo findhomograph
    // 现在的H矩阵是在中心为0的基础上投影变换的
    Mat H = findHomography(roi_corners,dst_corners);
//    Mat H_inv = findHomography(dst_corners,roi_corners);
//    cout<< "H:\n";
//    Get_Used_inv_H(H,corners,corners_trans);
//    cout<< "H_inv:\n";
//    Get_Used_inv_H(H_inv,corners,corners_trans);
    //todo 自定义perspective
    Mat dst;
    Mat H_pic = Center_WarpPerspective(H,img,dst,corners,1,"downloadpath");

//    vector<Mat> CornersForline;
//    vector<Point2d> CoPoint2d(4);
//    for(auto corn:corners){
//        cout << corn.t() << endl;
//
//        CornersForline.push_back(H_pic*corn);
//    }
//
//    for(auto corn:CornersForline){
////        cout << corn.t() << endl;
//        corn.at<double>(0,0) = corn.at<double>(0,0)/corn.at<double>(2,0);
//        corn.at<double>(1,0) = corn.at<double>(1,0)/corn.at<double>(2,0);
//        corn.at<double>(2,0) = corn.at<double>(2,0)/corn.at<double>(2,0);
//        cout << corn.t() << endl;
//    }
//    for (int i = 0; i < 4; ++i) {
//
//        CoPoint2d[i].x = CornersForline[i].at<double>(0,0);
//        CoPoint2d[i].y = CornersForline[i].at<double>(1,0);
//    }
//    line(dispimg, p1, p2, Scalar(0,0,255), 1);    //画直线

    return H_pic;
}



//todo 这里的H 是计算的H_inv
Mat Rotation::Center_WarpPerspective(
        const Mat H,
        const Mat src,Mat dst,
        const vector <Mat> corners,
        const int target,
        const string downloadpath){
    double fx = 800;
    double fy = 800;
    double Cx = src.cols/2;
    double Cy = src.rows/2;
    Mat write_pic;
    src.copyTo(write_pic);
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
    //    for (int i = 0; i < 4; ++i) {
    //        cout << "Cpoint_input " <<  Cpoint_input[i] << endl;
    //    }
    //todo 找到角坐标
    Point2d Zero_bias(0,0);
    Point2d OutputBorder(0,0);
    for (int i = 0; i < 4; ++i) {
    //        cout << "Cpoint_output " <<  Cpoint_output[i] << endl;
        if(Zero_bias.x > Cpoint_output[i].x)
            Zero_bias.x = Cpoint_output[i].x;
        if(Zero_bias.y > Cpoint_output[i].y)
            Zero_bias.y = Cpoint_output[i].y;
    }
    //    cout << Zero_bias << endl;
    for (int i = 0; i < 4; ++i) {
        Cpoint_output[i].x -= Zero_bias.x;
        Cpoint_output[i].y -= Zero_bias.y;
        Cpoint_output[i].x /= 1.5;
        Cpoint_output[i].y /= 1.5;
        cout << "Cpoint_output " <<  Cpoint_output[i] << endl;
        if(OutputBorder.x < Cpoint_output[i].x)
            OutputBorder.x = Cpoint_output[i].x;
        if(OutputBorder.y < Cpoint_output[i].y)
            OutputBorder.y = Cpoint_output[i].y;
    }
    //    cout <<"OutputBorder"<<OutputBorder;
    Mat H_pic = findHomography(Cpoint_input,Cpoint_output);
    return H_pic;

}

Mat Rotation::Get_Used_inv_H(const Mat H_input,const vector<Mat>corners,const vector<Mat>corners_trans){

    vector <Mat> show_corners(4);
    for (int i = 0; i < 4; ++i) {
        show_corners[i] = H_input*corners[i];
//        cout << "corner_trans after_H " <<  corners_trans[i].t() << endl;
    }
    for (int i = 0; i < 4; ++i) {

        show_corners[i] = H_input.inv()*corners_trans[i];
//        cout << "corner_trans after_Hinv " <<  corners_trans[i].t() << endl;
    }

    return H_input;
}

