//
// Created by aruchid on 2019/12/8.
//
#include "../include/aruchid_featuresfind.hpp"

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

// 像素坐标转相机归一化坐标

//void test()
//{
//    //TODO 这里是从图片匹配keypoint到图片映射H的整体匹配方法
//    //
//    // 注意这里改参数，原来是类参数
//
//
//
//    Mat img1, img2,img2_trans,img1_trans;
//    imgs[Rotated_].copyTo(img1);
//    imgs[Target_].copyTo(img2);
//
//    Rotate_angle MatVecs;
//    MatVecs.pitch = Eular_Angle[Rotated_].x - Eular_Angle[Target_].x;
//    MatVecs.yaw  = Eular_Angle[Rotated_].y - Eular_Angle[Target_].y;
//    MatVecs.roll   = Eular_Angle[Rotated_].z - Eular_Angle[Target_].z;
//    // offset
//    float h_offset2 = (1 - cos(rad(MatVecs.pitch ))) * img2.rows / 2;
//    float v_offset2 = (1 - cos(rad(MatVecs.yaw ))) * img2.cols / 2;
//    float NewWidth = img2.cols - v_offset2 * 2;
//    float NewHeight = img2.rows - h_offset2 * 2;
//    img2_trans = img2(Rect2f(v_offset2, h_offset2, NewWidth, NewHeight));
//    img1.copyTo(img1_trans);
//
////    img2.copyTo(img2_trans);
////    img1.copyTo(img1_trans);
//
//    Mat wrong;
//    float match_conf_ = 0.25f;
//    int num_matches_thresh1_ = 6;
//    int num_matches_thresh2_ = 6;
//    keypoints_1->clear();
//    keypoints_2->clear();
//
//    //todo 这里开始提取各类匹配点
//    //-- 初始化
//    Mat descriptors1_SIFT, descriptors2_SIFT;
//    vector<KeyPoint> keypoints1_SIFT, keypoints2_SIFT;
//
//    //todo use SIFT
//    Ptr<FeatureDetector> detectorSIFT = xfeatures2d::SIFT::create();
//    Ptr<DescriptorExtractor> descriptorSIFT = xfeatures2d::SIFT::create();
////    Ptr<FeatureDetector> detectorSIFT = AKAZE::create();
////    Ptr<DescriptorExtractor> descriptorSIFT = AKAZE::create();
//    // use this if you are in OpenCV2
//    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
//    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
//    // Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce" );
//
//    //-- 第一步:检测 Oriented FAST 角点位置
//    detectorSIFT->detect ( img1_trans,keypoints1_SIFT );
//    detectorSIFT->detect ( img2_trans,keypoints2_SIFT );
//
//    //-- 第二步:根据角点位置计算 BRIEF 描述子
//    descriptorSIFT->compute ( img1_trans, keypoints1_SIFT, descriptors1_SIFT );
//    descriptorSIFT->compute ( img2_trans, keypoints2_SIFT, descriptors2_SIFT );
//
//    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用KNN
//    vector<DMatch> matches;
//    //BFMatcher matcher ( NORM_HAMMING );
//
//    //FlannBasedMatcher matcher;
//    BFMatcher matcher;
//    vector<vector<DMatch> > matchePoints;
//    vector<DMatch> GoodMatchePoints;
//
//    cv::Mat src_des;
//    descriptors1_SIFT.copyTo(src_des);
//    vector<Mat> train_desc(1,src_des);
//    matcher.add(train_desc);
//    matcher.train();
//
//    cv::Mat dst_des;
//    descriptors2_SIFT.copyTo(dst_des);
//    matcher.knnMatch(dst_des, matchePoints, 2);
//
//    cout << "total match points: " << matchePoints.size() << endl;
//
//    // overlap comput
//    std::vector<Point2f> obj_pt, scene_pt;  //重复区域匹配点list
//    obj_pt.clear();
//    scene_pt.clear();
//
//    // Lowe's algorithm,获取优秀匹配点
//    for (int i = 0; i < matchePoints.size(); i++)
//    {
//        if (matchePoints[i][0].distance < 0.75 * matchePoints[i][1].distance)
//        {
//            GoodMatchePoints.push_back(matchePoints[i][0]);
//        }
//    }
//    cout << "Good match points: " << GoodMatchePoints.size() << std::endl;
//    std::vector<KeyPoint> match_src_keypoints;
//    match_src_keypoints.clear();
//    std::vector<KeyPoint> match_dst_keypoints;
//    match_dst_keypoints.clear();
//    for(int k = 0; k < GoodMatchePoints.size(); ++k)
//    {
//        obj_pt.push_back(keypoints1_SIFT[GoodMatchePoints[k].trainIdx].pt );
//        match_src_keypoints.push_back(keypoints1_SIFT[GoodMatchePoints[k].trainIdx]);
//        scene_pt.push_back(keypoints2_SIFT[GoodMatchePoints[k].queryIdx].pt);
//        match_dst_keypoints.push_back(keypoints2_SIFT[GoodMatchePoints[k].queryIdx]);
//    }
//
//    Mat inliers_mask;
//    Mat H = findHomography(scene_pt, obj_pt, RANSAC, 4.0, inliers_mask, 500, 0.9999);
////    std::cout << "H = " << H << std::endl;
////    std::cout << "H inv = " << H.inv() << std::endl;
//
//    return H;
//}

float rad(int x){
    return x * CV_PI / 180;
}

Mat FindMatch_CurseH_ (  Mat img1,Mat img2,
                        std::vector<KeyPoint>* keypoints_1,
                        std::vector<KeyPoint>* keypoints_2
)
{

    Mat img1_trans,img2_trans;
//todo rotated finished
    img2.copyTo(img2_trans);
    img1.copyTo(img1_trans);

    Mat wrong;
    float match_conf_ = 0.25f;
    int num_matches_thresh1_ = 6;
    int num_matches_thresh2_ = 6;
    keypoints_1->clear();
    keypoints_2->clear();

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
    detectorSIFT->detect ( img1_trans,keypoints1_SIFT );
    detectorSIFT->detect ( img2_trans,keypoints2_SIFT );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptorSIFT->compute ( img1_trans, keypoints1_SIFT, descriptors1_SIFT );
    descriptorSIFT->compute ( img2_trans, keypoints2_SIFT, descriptors2_SIFT );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用KNN
    vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    //    matcher->match ( descriptors_1, descriptors_2, match );
    KnnMatcher(descriptors1_SIFT, descriptors2_SIFT, matches, match_conf_);
    vector<Mat> imgdraw;
    imgdraw.push_back(img1);
    imgdraw.push_back(img2);

//    Mat dstKNN;
//    drawMatches(img1,keypoints1_SIFT,img2,keypoints2_SIFT,matches,dstKNN);
//    imwrite("./result/dstKNN.jpg",dstKNN);
//    matches = match;
//    cout << "//第四步:匹配点对筛选RANSAC"<< endl;

    //todo-- 第四步:匹配点对筛选RANSAC
    // Check if it makes sense to find homography
    //判断两幅图像的匹配点对的数量是否达到了设置的阈值，如果小于该阈值，说明两幅图像没有重叠的地方，无需再进行拼接
    if (matches.size() < static_cast<size_t>(num_matches_thresh1_))
        return wrong;

    Size img_size = Size(img1.cols,img1.rows);
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
    H_return = findHomography(src_points, dst_points, RANSAC, 4.0, inlier_mask, 500, 0.9999);

//    Draw_match(keypoints1_SIFT,keypoints2_SIFT,matches,imgdraw,inlier_mask);
    int num_inliers = 0;    //匹配点对的内点数先清零
    //由内点掩码得到内点数
    for (size_t i = 0; i < inlier_mask.size(); ++i)
        if (inlier_mask[i]){
            num_inliers++;}
    cout << "num_inliers = " <<num_inliers <<endl;
    int inlier_idx = 0;    //表示内点索引
    for (size_t i = 0; i < matches.size(); ++i)    //遍历匹配点对，得到内点
    {
        if (!inlier_mask[i])    //不是内点
            continue;
        const DMatch &m = matches[i];    //赋值

        Point2f p = keypoints1_SIFT[m.queryIdx].pt;    //第一幅图像的内点坐标
        src_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoints_1->push_back(keypoints1_SIFT[m.queryIdx]);

        p = keypoints2_SIFT[m.trainIdx].pt;    //第二幅图像的内点坐标
        dst_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoints_2->push_back(keypoints2_SIFT[m.trainIdx]);
        inlier_idx++;    //索引计数
//        matches_output->push_back(matches[i]);
    }


    Mat meanPoint,std;
    meanStdDev(src_points,meanPoint,std);
    cout << "meanPoint = " << meanPoint<< endl;
    cout << "std = " << std << endl;
    Point2d meanp(meanPoint.at<double>(0,0),meanPoint.at<double>(1,0));

    circle(img1,meanp,10,Scalar(0,240,0),10);
    imwrite("img1.jpg",img1);

    meanStdDev(dst_points,meanPoint,std);
    cout << "meanPoint = " << meanPoint<< endl;
    cout << "std = " << std << endl;
    Point2d meanp2(meanPoint.at<double>(0,0),meanPoint.at<double>(1,0));

    cout << meanp-meanp2 << endl;
    circle(img2,meanp2,10,Scalar(0,240,0),10);
    imwrite("img2.jpg",img2);

    H_return = findHomography(src_points, dst_points, 0);
//    return mean;
}


Mat FindMatch_CurseH (  vector<Mat> imgs,
                    int Rotated_,
                    int Target_,
                    std::vector<KeyPoint>* keypoints_1,
                    std::vector<KeyPoint>* keypoints_2,
                    vector<Point3d> Eular_Angle,
                    int Best_POV
                    )
{
    //TODO 这里是从图片匹配keypoint到图片映射H的整体匹配方法
    //
    // 注意这里改参数，原来是类参数

    Mat img1, img2,img2_trans,img1_trans;
    imgs[Rotated_].copyTo(img1);
    imgs[Target_].copyTo(img2);

    resize(img1,img1,Size(img1.cols/4,img1.rows/4));
    resize(img2,img2,Size(img2.cols/4,img2.rows/4));

    Rotate_angle MatVecs;
    MatVecs.pitch = Eular_Angle[Rotated_].x - Eular_Angle[Target_].x;
    MatVecs.yaw  = Eular_Angle[Rotated_].y - Eular_Angle[Target_].y;
    MatVecs.roll   = Eular_Angle[Rotated_].z - Eular_Angle[Target_].z;
    MatVecs.roll = MatVecs.roll*(-1);
//    MatVecs.roll   = -15;

//    cout <<  MatVecs.pitch <<"\t"<< MatVecs.yaw <<"\t"<< MatVecs.roll << endl;
    // offset
    float h_offset2 = (1 - cos(rad(MatVecs.pitch ))) * img2.rows / 2;
    float v_offset2 = (1 - cos(rad(MatVecs.yaw ))) * img2.cols / 2;
    float NewWidth = img2.cols - v_offset2 * 2;
    float NewHeight = img2.rows - h_offset2 * 2;
//    cout << "cos(60) " << cos(rad(300)) << endl;
//    cout << "h_offset2 " << h_offset2 << endl;
//    cout << "v_offset2 " << v_offset2 << endl;
//    cout << "NewHeight " << NewHeight << endl;
//    cout << "NewWidth " << NewWidth << endl;
    img1.copyTo(img1_trans);
    img2.copyTo(img2_trans);

    //todo 写rotation
//    float  x,y,RotWitdh,RotHeight;
//    float tanzeta = tan(rad(MatVecs.roll));
////    tanzeta = tan(rad(10));
//    cout << "tanzeta " << tanzeta <<endl;
//    y = (NewHeight-NewWidth*tanzeta)/(1-tanzeta*tanzeta);
//    x = (NewWidth-NewHeight*tanzeta)/(1-tanzeta*tanzeta);
//    cout << "y " << y<< endl;
//    cout << "x " << x<< endl;
//    cout << "cos(rad(MatVecs.roll)) " << cos(MatVecs.roll) << endl;
//    RotWitdh = x/cos(rad(MatVecs.roll));
//    RotHeight = y/cos(rad(MatVecs.roll));
//    cout << "RotWitdh " << RotWitdh<< endl;
//    cout << "RotHeight " << RotHeight<< endl;
//    Mat mask = Mat(Size(RotWitdh, RotHeight), CV_8UC1, Scalar(255, 255, 255));
////    mask = mask(Rect2f(v_offset2, h_offset2, NewWidth, NewHeight));
//    cout << mask.size<< endl;
//    //旋转
//    Point2f center( (float)(RotWitdh/2) , (float) (RotHeight/2));
//    Mat affine_matrix = getRotationMatrix2D( center, MatVecs.roll, 1 );//求得旋转矩阵
////    cout << endl
//
//    affine_matrix.at<double>(0,2) += (img1.cols-mask.cols)/2;
//    affine_matrix.at<double>(1,2) += (img1.rows-mask.rows)/2;
//
//    Mat mask_dst = Mat(Size(img1.size()), CV_8UC1, Scalar(255, 255, 255));
//    warpAffine(mask, mask_dst, affine_matrix, img1.size());
//    imwrite("./result/result_mask.jpg",mask_dst);

    Mat mask = Mat(Size(NewWidth, NewHeight), CV_8UC1, Scalar(255, 255, 255));
//    mask = mask(Rect2f(v_offset2, h_offset2, NewWidth, NewHeight));
//    cout << mask.size<< endl;
    //旋转
    Point2f center( (float)(NewWidth/2) , (float) (NewHeight/2));
    Mat affine_matrix = getRotationMatrix2D( center, MatVecs.roll, 1 );//求得旋转矩阵
//    cout << endl

    affine_matrix.at<double>(0,2) += (img1.cols-mask.cols)/2;
    affine_matrix.at<double>(1,2) += (img1.rows-mask.rows)/2;

    Mat mask_dst = Mat(Size(img1.size()), CV_8UC1, Scalar(255, 255, 255));
//    warpAffine(mask, mask_dst, affine_matrix, img1.size());
//    imwrite("./result/result_mask.jpg",mask_dst);


//todo rotated finished
//    img2.copyTo(img2_trans);
//    img1.copyTo(img1_trans);

    Mat wrong;
    float match_conf_ = 0.25f;
    int num_matches_thresh1_ = 6;
    int num_matches_thresh2_ = 6;
    keypoints_1->clear();
    keypoints_2->clear();

    //todo 这里开始提取各类匹配点
    //-- 初始化
    Mat descriptors1_SIFT, descriptors2_SIFT;
    vector<KeyPoint> keypoints1_SIFT, keypoints2_SIFT;

    //todo use SIFT
    Ptr<FeatureDetector> detectorSIFT = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> descriptorSIFT = xfeatures2d::SIFT::create();
//    Ptr<FeatureDetector> detectorSIFT = AKAZE::create();
//    Ptr<DescriptorExtractor> descriptorSIFT = AKAZE::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    // Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce" );

    //-- 第一步:检测 Oriented FAST 角点位置
    detectorSIFT->detect ( img1_trans,keypoints1_SIFT ,mask_dst);
    detectorSIFT->detect ( img2_trans,keypoints2_SIFT );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptorSIFT->compute ( img1_trans, keypoints1_SIFT, descriptors1_SIFT );
    descriptorSIFT->compute ( img2_trans, keypoints2_SIFT, descriptors2_SIFT );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用KNN
    vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
//    matcher->match ( descriptors_1, descriptors_2, match );
    KnnMatcher(descriptors1_SIFT, descriptors2_SIFT, matches, match_conf_);
    vector<Mat> imgdraw;
    imgdraw.push_back(img1);
    imgdraw.push_back(img2);

//    Mat dstKNN;
//    drawMatches(img1,keypoints1_SIFT,img2,keypoints2_SIFT,matches,dstKNN);
//    imwrite("./result/dstKNN.jpg",dstKNN);
//    matches = match;
//    cout << "//第四步:匹配点对筛选RANSAC"<< endl;

    //todo-- 第四步:匹配点对筛选RANSAC
    // Check if it makes sense to find homography
    //判断两幅图像的匹配点对的数量是否达到了设置的阈值，如果小于该阈值，说明两幅图像没有重叠的地方，无需再进行拼接
    if (matches.size() < static_cast<size_t>(num_matches_thresh1_))
        return wrong;

    Size img_size = Size(img1.cols,img1.rows);
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
        keypoints_1->push_back(keypoints1_SIFT[m.queryIdx]);

        p = keypoints2_SIFT[m.trainIdx].pt;    //第二幅图像的内点坐标
        dst_points.at<Point2f>(0, inlier_idx) = p;    //赋值
        keypoints_2->push_back(keypoints2_SIFT[m.trainIdx]);
        inlier_idx++;    //索引计数
//        matches_output->push_back(matches[i]);
    }

    H_return = findHomography(src_points, dst_points, 0);
    return H_return;
}

void KnnMatcher(const Mat descriptors_1, const Mat descriptors_2,
                vector<DMatch>& Dmatchinfo,const float match_conf_)
{
    Dmatchinfo.clear();    //清空
    //定义K-D树形式的索引
    Ptr<flann::IndexParams> indexParams = new flann::KDTreeIndexParams();
    //定义搜索参数
    Ptr<flann::SearchParams> searchParams = new flann::SearchParams();

    if (descriptors_1.depth() == CV_8U)
    {
        indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
        searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
    }
    //使用FLANN方法匹配，定义matcher变量
    FlannBasedMatcher matcher(indexParams, searchParams);
    vector< vector<DMatch> > pair_matches;    //表示邻域特征点
    MatchesSet matches;    //表示匹配点对

    // Find 1->2 matches
    //在第二幅图像中，找到与第一幅图像的特征点最相近的两个特征点
    matcher.knnMatch(descriptors_1, descriptors_2, pair_matches, 2);
    for (size_t i = 0; i < pair_matches.size(); ++i)    //遍历这两次匹配结果
    {
        //如果相近的特征点少于2个，则继续下个匹配
        if (pair_matches[i].size() < 2)
            continue;
        //得到两个最相近的特征点
        const DMatch& m0 = pair_matches[i][0];
        const DMatch& m1 = pair_matches[i][1];
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
    //在第一幅图像中，找到与第二幅图像的特征点最相近的两个特征点
//    matcher.knnMatch(descriptors_2, descriptors_1, pair_matches, 2);
//    for (size_t i = 0; i < pair_matches.size(); ++i)    //遍历这两次匹配结果
//    {
//        //如果相近的特征点少于2个，则继续下个匹配
//        if (pair_matches[i].size() < 2)
//            continue;
//        //得到两个最相近的特征点
//        const DMatch& m0 = pair_matches[i][0];
//        const DMatch& m1 = pair_matches[i][1];
//        if (m0.distance < (1.f - match_conf_) * m1.distance)    //表明匹配成功，式1
//            //如果当前的匹配点对还没有被上一次调用knnMatch函数时得到，则需要把这次的匹配点对保存下来
//            if (matches.find(make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
//                Dmatchinfo.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
//    }
    //    cout << "rough matching..." << endl;
//    LOG("1->2 & 2->1 matches: " << Dmatchinfo.size() << endl);
}



Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
            (
                    ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
                    ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
            );
}




void pose_estimation_2d2d ( std::vector<KeyPoint> keypoints_1,
                            std::vector<KeyPoint> keypoints_2,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t ,
                            std::vector< char > inlier_mask)
{
    // 相机内参,TUM Freiburg2
//    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    Mat K = ( Mat_<double> ( 3,3 ) << 888, 0, 360, 0, 888, 840, 0, 0, 1 );

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    vector<uchar> inlier_mask_fake;
//    vector<char> inlier_Mask_trans;
    fundamental_matrix = findFundamentalMat ( points1, points2,inlier_mask_fake, FM_RANSAC );
//    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;
//    cout << inlier_mask_fake.size() << endl;
    for (int i = 0; i < inlier_mask_fake.size(); ++i) {
//        cout << std::hex << static_cast<unsigned short>(inlier_mask[i]) ;
        inlier_mask.push_back(static_cast<char>(inlier_mask_fake[i]));
//        cout << "  "<<std::hex << static_cast<unsigned short>(inlier_mask[i]) << endl;
    }

    //-- 计算本质矩阵
//    Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值
//    double focal_length = 521;			//相机焦距, TUM dataset标定值

    Point2d principal_point ( 360, 840 );	//相机光心, TUM dataset标定值
    double focal_length = 800;			//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
//    cout<<"R is "<<endl<<R<<endl;
//    cout<<"t is "<<endl<<t<<endl;


}
