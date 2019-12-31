

bool set_src_feature(cv::Mat &src, ImageFeatures &src_features){

    if (src.empty())
    {
        std::cout << "src is empty." << std::endl;
        return false;
    }
    double overlap_input_megapix = 0.2;  //输入图片默认尺寸
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


bool overlap_point(cv::Mat &dst, ImageFeatures &src_features ,vector<Point> &src_pnts, vector<Point> &dst_pnts){


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

    double overlap_input_megapix = 0.2;  //输入图片默认尺寸
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

    if (dst_features.keypoints.size() < 200)
    {
        dst_fastdetector.release();
        std::cout << "dst feature kp size < 200. " << std::endl;
        return false;
    }

    KeyPointsFilter::runByImageBorder(dst_features.keypoints, overlap_input.size(), 31);
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
    tree = makePtr<cv::flann::Index>(src_features.descriptors, cv::flann::LshIndexParams(5, 15, 0), cvflann::FLANN_DIST_HAMMING);

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

    cout << "obj_pt" << endl;
    cout << obj_pt.size() << endl;
    cout << scene_pt.size() << endl;

    if (obj_pt.size() < 50)
    {
        dst_extractor.release();
        dst_fastdetector.release();
        flann_search_dst.release();
        std::cout << "obj_pt.size() < 50. " << std::endl;
        return false;
    }
    //compute warping matrix
    std::vector<uchar> inliers_mask;
    cv::Mat H = estimateAffine2D(obj_pt, scene_pt, inliers_mask);
    H.push_back(Mat::zeros(1, 3, CV_64F));
    H.at<double>(2, 2) = 1;

    int good_num = 0;
    for (int i=0; i<inliers_mask.size();++i){
        if (inliers_mask[i] != '\0')
            good_num++;
    }

    float conf = good_num /(8 + 0.3 * (obj_pt.size()));
    if (good_num < 10 || conf < 0.5)
//    if (good_num < inliner_num || conf < conf_thresh)
    {
        dst_extractor.release();
        dst_fastdetector.release();
        flann_search_dst.release();
        std::cout << "good_num < inliner_num or conf < conf_thresh" << std::endl;
        return false;
    }

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)overlap_input.cols, 0 );
    obj_corners[2] = Point2f( (float)overlap_input.cols, (float)overlap_input.rows );
    obj_corners[3] = Point2f( 0, (float)overlap_input.rows );

    std::vector<Point2f> scene_corners(4);

    //dst points transformation
    perspectiveTransform(obj_corners, scene_corners, H);
    //
    //    if (abs(scene_corners[0].x-scene_corners[1].x) > 4*abs(obj_corners[0].x-obj_corners[1].x) ||
    //        abs(scene_corners[2].x-scene_corners[3].x) > 4*abs(obj_corners[2].x-obj_corners[3].x) ||
    //        abs(scene_corners[1].y-scene_corners[2].y) > 4*abs(obj_corners[1].y-obj_corners[2].y) ||
    //        abs(scene_corners[3].y-scene_corners[0].y) > 4*abs(obj_corners[3].y-obj_corners[0].y) )
    //
    //        return false;

    //upsample
    float scale = 1. / overlap_input_scale;
    for(int i=0; i<4; i++){
        dst_pnts[i].x = scene_corners[i].x*scale;
        dst_pnts[i].y = scene_corners[i].y*scale;

        // 边界保护
        //if (dst_pnts[i].x < 0) dst_pnts[i].x = 0;
        //if (dst_pnts[i].y < 0) dst_pnts[i].y = 0;
        //if (dst_pnts[i].x > dst.cols) dst_pnts[i].x = dst.cols;
        //if (dst_pnts[i].y > dst.rows) dst_pnts[i].y = dst.rows;
    }
    //src points transformation
    perspectiveTransform( obj_corners, scene_corners, H.inv());
    //upsample
    for(int i=0; i<4; i++){
        src_pnts[i].x = scene_corners[i].x*scale;
        src_pnts[i].y = scene_corners[i].y*scale;

        // 边界保护
        // if (src_pnts[i].x < 0) src_pnts[i].x = 0;
        // if (src_pnts[i].y < 0) src_pnts[i].y = 0;
        // if (src_pnts[i].x > dst.cols) src_pnts[i].x = dst.cols;
        // if (src_pnts[i].y > dst.rows) src_pnts[i].y = dst.rows;
    }
    gettimeofday( &match_end, NULL );
    //求出两次时间的差值，单位为us
    int matchtimeuse = 1000000 * ( match_end.tv_sec - match_start.tv_sec ) + match_end.tv_usec - match_start.tv_usec;
//    std::cout << "match time is " << matchtimeuse << "  us."<< std::endl;
    dst_extractor.release();
    dst_fastdetector.release();
    flann_search_dst.release();

    LOGLN("feature get, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    return true;
}
//
// Created by aruchid on 2019/12/19.
//

