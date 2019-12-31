//
// Created by aruchid on 2019/12/8.
//

#include "../include/aruchid_printpic.h"
//#include "../include/aruchid_log.h"
#include "opencv2/stitching/detail/util.hpp"
void Projection::CalcCorners(const Mat& H, const Mat& src, vector<Point2f> &corners)
{
    Point2f left_top, left_bottom, right_top, right_bottom;

    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2); //列向量
    Mat V1 = Mat(3, 1, CV_64FC1, v1); //列向量

    V1 = H * V2;
//左上角(0,0,1)
//    cout << "V2: " << V2 << endl;
    // cout << "V1: " << v1[2] << endl;
    left_top.x = v1[0] / v1[2];
    left_top.y = v1[1] / v1[2];
    corners.push_back(left_top);

//左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2); //列向量
    V1 = Mat(3, 1, CV_64FC1, v1); //列向量
    V1 = H * V2;
    left_bottom.x = v1[0] / v1[2];
    left_bottom.y = v1[1] / v1[2];
    corners.push_back(left_bottom);

//右上角(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2); //列向量
    V1 = Mat(3, 1, CV_64FC1, v1); //列向量
    V1 = H * V2;
    right_top.x = v1[0] / v1[2];
    right_top.y = v1[1] / v1[2];
    corners.push_back(right_top);

//右下角(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2); //列向量
    V1 = Mat(3, 1, CV_64FC1, v1); //列向量
    V1 = H * V2;
    right_bottom.x = v1[0] / v1[2];
    right_bottom.y = v1[1] / v1[2];
    corners.push_back(right_bottom);
}
Mat mywarpPerspective(Mat src,Mat T){
//    namedWindow(",",WINDOW_NORMAL)
    //此处注意计算模型的坐标系与Mat的不同
    //图像以左上点为（0,0），向左为x轴，向下为y轴，所以前期搜索到的特征点 存的格式是（图像x，图像y）---（rows，cols）
    //而Mat矩阵的是向下为x轴，向左为y轴，所以存的方向为（图像y，图像x）----（cols，rows）----（width，height）
    //这个是计算的时候容易弄混的
    //创建原图的四个顶点的3*4矩阵（此处我的顺序为左上，右上，左下，右下）

    Mat tmp(3, 4, CV_64FC1, 1);
    tmp.at < double >(0, 0) = 0;
    tmp.at < double >(1, 0) = 0;
    tmp.at < double >(0, 1) = src.cols;
    tmp.at < double >(1, 1) = 0;
    tmp.at < double >(0, 2) = 0;
    tmp.at < double >(1, 2) = src.rows;
    tmp.at < double >(0, 3) = src.cols;
    tmp.at < double >(1, 3) = src.rows;
//    cout << tmp << endl;

    //获得原图四个顶点变换后的坐标，计算变换后的图像尺寸
    Mat corner = T * tmp;      //corner=(x,y)=(cols,rows)

    auto lt_x = (float)(corner.at < double >(0, 0) / corner.at < double >(2,0));
    auto lt_y = (float)(corner.at < double >(1, 0) / corner.at < double >(2,0));
    auto rt_x = (float)(corner.at < double >(0, 1) / corner.at < double >(2,1));
    auto rt_y = (float)(corner.at < double >(1, 1) / corner.at < double >(2,1));
    auto lb_x = (float)(corner.at < double >(0, 2) / corner.at < double >(2,2));
    auto lb_y = (float)(corner.at < double >(1, 2) / corner.at < double >(2,2));
    auto rb_x = (float)(corner.at < double >(0, 3) / corner.at < double >(2,3));
    auto rb_y = (float)(corner.at < double >(1, 3) / corner.at < double >(2,3));

//    std::cout << "lt_x = " << lt_x << ", lt_y = " << lt_y << std::endl;
//    std::cout << "lb_x = " << lb_x << ", lb_y = " << lb_y << std::endl;
//    std::cout << "rt_x = " << rt_x << ", rt_y = " << rt_y << std::endl;
//    std::cout << "rb_x = " << rb_x << ", rb_y = " << rb_y << std::endl;

    int width = 0, height = 0;
    double maxw = corner.at < double >(0, 0)/ corner.at < double >(2,0);
    double minw = corner.at < double >(0, 0)/ corner.at < double >(2,0);
    double maxh = corner.at < double >(1, 0)/ corner.at < double >(2,0);
    double minh = corner.at < double >(1, 0)/ corner.at < double >(2,0);

    lt_x = lt_x + maxw;
    lt_y = lt_y - maxh;
    lb_x = lb_x + maxw;
    lb_y = lb_y - maxh;
    rt_x = rt_x + maxw;
    rt_y = rt_y - maxh;
    rb_x = rb_x + maxw;
    rb_y = rb_y - maxh;

//    std::cout << "corner = " << corner << std::endl;
//    std::cout << "maxw = " << maxw << ", minw = " << minw << ", maxh = " << maxh << ",minh = " << minh << std::endl;
    for (int i = 1; i < 4; i++) {
//        cout << "maxmin tt" << endl;
        maxw = max(maxw, corner.at < double >(0, i) / corner.at < double >(2, i));
        minw = min(minw, corner.at < double >(0, i) / corner.at < double >(2, i));
        maxh = max(maxh, corner.at < double >(1, i) / corner.at < double >(2, i));
        minh = min(minh, corner.at < double >(1, i) / corner.at < double >(2, i));
    }
    //创建向前映射矩阵 map_x, map_y

    Mat dst;
    //size(height,width)
    dst.create(int(maxh - minh), int(maxw - minw), src.type());
//    cout << "dst.create()"<<endl;
//    std::cout << "height = " << maxh - minh << ", width = " << maxw - minw << std::endl;
    Mat map_x(dst.size(), CV_32FC1);
    Mat map_y(dst.size(), CV_32FC1);

    Mat proj(3,1, CV_32FC1,1);
    Mat point(3,1, CV_32FC1,1);

    T.convertTo(T, CV_32FC1);

    //本句是为了令T与point同类型（同类型才可以相乘，否则报错，也可以使用T.convertTo(T, point.type() );）
    Mat Tinv = T.inv();

    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            point.at<float>(0) = j + minw ;
            point.at<float>(1) = i + minh ;
            proj = Tinv * point;
            map_x.at<float>(i, j) = proj.at<float>(0) / proj.at<float>(2);
            map_y.at<float>(i, j) = proj.at<float>(1) / proj.at<float>(2);
        }
    }

    remap(src,dst,map_x,map_y, INTER_LINEAR);
//    imwrite("./result/dst.jpg",dst);
//    dst.copyTo(output);
    return dst;
}

void Project_region_Get(Mat img,Mat H_trans,Mat &Translation_all,Size &OutputSize){

    cout << "Project_region_Get" << endl;
    vector<Point2f> corners_Rotated(4);
    vector<Mat> corners_Rotated_homogeneous;
    vector<Mat> corners_after_homogeneous(4);
    corners_Rotated[left_top]= Point2f(0,0);
    corners_Rotated[right_top]= Point2f(img.cols,0);
    corners_Rotated[left_bottom]= Point2f(0,img.rows);
    corners_Rotated[right_bottom]= Point2f(img.cols,img.rows);

    cout << H_trans << endl;
    for (int i = 0; i < 4; ++i) {

//        cout << "for 1" << endl;
        Mat temp_corner =(cv::Mat_<double>(1, 3) << (double)corners_Rotated[i].x,(double)corners_Rotated[i].y,1);

//        corners_Rotated_homogeneous[i].at<double>(0,0) = corners_Rotated[i].x;
//        corners_Rotated_homogeneous[i].at<double>(0,1) = corners_Rotated[i].y;
//        corners_Rotated_homogeneous[i].at<double>(0,2) = 1;
        corners_Rotated_homogeneous.push_back(temp_corner);
        cout << corners_Rotated_homogeneous[i] << endl;
        corners_after_homogeneous[i] = (H_trans * corners_Rotated_homogeneous[i].t()).t();
        cout << corners_after_homogeneous[i] << endl;
    }
    vector <Point2f> Project_corners(2);
    Project_corners[0] = Point2f(0,0);
    Project_corners[1] = Point2f(img.cols,img.rows);


    for (int i = 0; i < 4; ++i) {
        if(Project_corners[0].x > corners_after_homogeneous[i].at<double>(0,0))
            Project_corners[0].x = corners_after_homogeneous[i].at<double>(0,0);
        if(Project_corners[0].y > corners_after_homogeneous[i].at<double>(0,1))
            Project_corners[0].y = corners_after_homogeneous[i].at<double>(0,1);
        if(Project_corners[1].x < corners_after_homogeneous[i].at<double>(0,0))
            Project_corners[1].x = corners_after_homogeneous[i].at<double>(0,0);
        if(Project_corners[1].y < corners_after_homogeneous[i].at<double>(0,1))
            Project_corners[1].y = corners_after_homogeneous[i].at<double>(0,1);
    }
    Translation_all =(cv::Mat_<double>(3, 3) << 0,0,-Project_corners[0].x,0,0,-Project_corners[0].y,0,0,0);
    cout << Translation_all << endl;
//    Trans_temp.copyTo(Translation_all);
    Size ProjectionSize = Size(
            Project_corners[1].x - Project_corners[0].x,
            Project_corners[1].y - Project_corners[0].y);
    OutputSize = ProjectionSize;
}

Projection::Projection(){
    PanoSize = Size( 0,0);
    BestPOV = 0;
}
Projection::~Projection(){
    PanoSize = Size( 0,0);
    BestPOV = 0;
    Imgs.clear();
    Imgs_Projected.clear();//投影以后的图片,扭转+黑边
    Homo.clear();//输入进来的两两之间的单应性矩阵
    Homo_Projected.clear();//相对于BestPOV 两两之间累加的单应性矩阵
    Trans.clear(); //平移矩阵
    mask.clear();  //每张图片的mask
    SinglePrjSize.clear();//
    CornersInPano.clear();
    corners_single.clear();
}



bool Projection::InfoStack(vector<Mat> Imgs_input,vector<Mat> Homo_input,int BestPOV_input){
    
    BestPOV = BestPOV_input;//载入最佳变换目标
    Imgs.swap(Imgs_input);//图片positon copyto Imgs
//    cout << "infostack "<< Imgs.size() << endl;
//    imwrite("./testInforstack.jpg",Imgs[0]);
    Homo_Projected.clear();
	// cout << RED <<"InfoStack get in" << endl;
    PicNums_ = Imgs.size();
    Mat tempEye = Mat::eye(3,3,CV_64F);
	// cout << tempEye << endl;
    // 放入Homo,HomoProjected
    
    for (int j = 0; j < Homo_input.size(); ++j) {
        Homo.push_back(Homo_input[j]);
    }
    
    for (int picnum = 0; picnum < PicNums_;) {
	//        cout <<  "turn = " << picnum << endl;
        if(picnum < BestPOV){
	//            cout << BLUE << "Homo projected NUm :";
	//            cout << picnum << GREEN << endl;
            Mat tempHomo;
            tempEye.copyTo(tempHomo);
            for (int step = picnum+1; step <= BestPOV ;) {
//                cout << step ;
                tempHomo = Homo[step].inv() * tempHomo;
	//                cout <<"projected = \n"<< Homo_Projected[picnum] << endl;
                step++;
            }
            Homo_Projected.push_back(tempHomo);
        }
        else if (picnum > BestPOV){
	//            cout << BLUE << "Homo projected NUm :" << picnum << GREEN<< endl;
            Mat tempHomo;
            tempEye.copyTo(tempHomo);
            for (int step = BestPOV+1; step <= (picnum);) {
//                cout << step ;
                tempHomo =  tempHomo * Homo[step];
//                cout <<"projected = \n"<< Homo_Projected[picnum] << endl;
                step++;
            }
            Homo_Projected.push_back(tempHomo);
        }
        else if (picnum == BestPOV){
//            cout << BLUE << "Homo projected NUm :" << picnum << GREEN<< endl;
            Mat tempHomo;//深拷贝问题
            tempEye.copyTo(tempHomo);
//            cout << tempEye << endl;
            Homo_Projected.push_back(tempHomo);
//            cout <<"projected centre = \n"<< Homo_Projected[picnum] << endl;
        }
        picnum++;

    }
//    PicNums_ = Imgs.size();
//    cout << "Homo_Projected = " << BestPOV << endl;
//    cout << "Homo_Projected = " << Homo_Projected.size() << endl;
//    for (int i = 0; i < PicNums_ ; ++i) {
//        cout << Homo_Projected[i] << endl;
//    }

//    for (int i = 0; i < PicNums_ - 1; ++i) {
//        cout << Homo[i] << endl;
//    }
//    cout <<RED<< "InfoStack get out" << RESET << endl;
};

void Projection::Img_FieldTrans_Find(){
	
//	cout << "Imgs[i].size" << endl;
    vector<vector<Point2f>> corners(PicNums_);
//	cout << "Imgs[].size" << endl;
    for (int i = 0; i < PicNums_; ++i) {
//        cout << PicNums_ << endl;
//        cout << Imgs_Projected[i].size << endl;
//        cout << Homo[i] << endl;
        //todo 把所有图片转换成投影变换后的图片(非全局)
        Mat temp_projected = mywarpPerspective(Imgs[i], Homo_Projected[i]);
//        cout << temp_projected.size << endl;
        Imgs_Projected.push_back(temp_projected);
//        imwrite("./result/imgprojected_"+to_string(i)+".jpg",Imgs_Projected[i]);
        vector<Point2f> tempcorners;
        CalcCorners(Homo_Projected[i], Imgs[i], corners[i]);
//        cout << "corners " << corners[i] << endl;
//        cout << "CalcCorner successed " << corners[i]<< endl;
//        tempcorners.copyTo(corners[i]);

        //TODO CalcCorners 计算的是原始图片四个顶点在warp后的图像位置
        // 这里需要改成四个顶点的外接矩形坐标
        Point tmp0, tmp1, LT_position;
        //LT_position 是该图片在全景图中的左上角位置坐标
        tmp0.x = min(corners[i][0].x, corners[i][1].x);
        tmp1.x = min(corners[i][2].x, corners[i][3].x);
        LT_position.x = min(tmp0.x, tmp1.x);

        tmp0.y = min(corners[i][0].y, corners[i][1].y);
        tmp1.y = min(corners[i][2].y, corners[i][3].y);
        LT_position.y = min(tmp0.y, tmp1.y);
        //SinglePrjSize 尺寸
        //CornersInPano 全局位置
        SinglePrjSize.push_back(Imgs_Projected[i].size());
        CornersInPano.push_back(LT_position);//corners.left_top;
    }
    corners_single = corners;

}

Size Projection::Img_FieldScale_Find(){

//	cout << "Imgs[i].size" << endl;
    vector<vector<Point2f>> corners(PicNums_);
//	cout << "Imgs[].size"  << PicNums_<< endl;
    for (int i = 0; i < PicNums_; ++i) {
//        cout << PicNums_ << endl;
//        cout << Imgs_Projected[i].size << endl;
//        cout << Homo[i] << endl;
//        Mat temp_projected = mywarpPerspective(Imgs[i], Homo_Projected[i]);
//////        cout << temp_projected.size << endl;
//        Imgs_Projected.push_back(temp_projected);
//        imwrite("./result/imgprojected_"+to_string(i)+".jpg",Imgs_Projected[i]);
        vector<Point2f> tempcorners;
        CalcCorners(Homo_Projected[i], Imgs[i], corners[i]);
//        cout << "Imgs[].CalcCorners"  << Homo_Projected.size()<< endl;
//        cout << "Imgs[].CalcCorners"  << Imgs.size()<< endl;
//        cout << "corners " << corners[i] << endl;
//        cout << "CalcCorner successed " << corners[i]<< endl;
//        tempcorners.copyTo(corners[i]);

        //TODO CalcCorners 计算的是原始图片四个顶点在warp后的图像位置
        // 这里需要改成四个顶点的外接矩形坐标
        Point tmp0, tmp1, LT_position,RB_position;
        //LT_position 是该图片在全景图中的左上角位置坐标
        tmp0.x = min(corners[i][0].x, corners[i][1].x);
        tmp1.x = min(corners[i][2].x, corners[i][3].x);
        LT_position.x = min(tmp0.x, tmp1.x);

        tmp0.y = min(corners[i][0].y, corners[i][1].y);
        tmp1.y = min(corners[i][2].y, corners[i][3].y);
        LT_position.y = min(tmp0.y, tmp1.y);
        //SinglePrjSize 尺寸
        tmp0.x = max(corners[i][0].x, corners[i][1].x);
        tmp1.x = max(corners[i][2].x, corners[i][3].x);
        RB_position.x = max(tmp0.x, tmp1.x) - LT_position.x;

        tmp0.y = max(corners[i][0].y, corners[i][1].y);
        tmp1.y = max(corners[i][2].y, corners[i][3].y);
        RB_position.y = max(tmp0.y, tmp1.y) - LT_position.y;
        //CornersInPano 全局位置

        SinglePrjSize.push_back(Size(RB_position.x, RB_position.y));
//        cout << "SinglePrjSize.push_back" << Imgs_Projected[i].size() << endl;
//        cout << "LT_position.push_back" << LT_position << endl;
//        cout << "RB_position.push_back" << RB_position << endl;
        CornersInPano.push_back(LT_position);//corners.left_top;
    }
//    cout << "Ioop succ"  << PicNums_<< endl;
    corners_single = corners;
    Rect dst_roi = resultRoi(CornersInPano, SinglePrjSize);
    return dst_roi.size();
}



//todo 找到每一张图的投影位置
void Projection::Img_FieldTrans_Find(bool tar){
    //todo 初始化
//    cout << "Img_FieldTrans_Find" << endl;
    Size ImgSize_before = Size(Imgs[0].cols,Imgs[0].rows);
//        vector <Point2f> Project_corners(2);
    CornersInPano[0] = Point2f(0,0);//最左上角
    CornersInPano[1] = Point2f(ImgSize_before.width,ImgSize_before.height);//最右下角

    vector<Point2f> corners_Before_projected(4);
    corners_Before_projected[left_top]= Point2f(0,0);
    corners_Before_projected[right_top]= Point2f(ImgSize_before.width,0);
    corners_Before_projected[left_bottom]= Point2f(0,ImgSize_before.height);
    corners_Before_projected[right_bottom]= Point2f(ImgSize_before.width,ImgSize_before.height);


    //todo 根据图片顺序，依次投影
    for (int picnum = 0; picnum < Imgs.size(); ++picnum) {
//        cout << "Project_region_Get" << endl;

        vector<Mat> corners_before_homogeneous;
        vector<Mat> corners_after_homogeneous(4);
        for (int i = 0; i < 4; ++i) {
            //todo 二维坐标 转为 齐次坐标
//            cout << "for 1 step" << endl;
            Mat temp_corner =(cv::Mat_<double>(1, 3) << corners_Before_projected[i].x,corners_Before_projected[i].y,1);
            corners_before_homogeneous.push_back(temp_corner);
//            cout << corners_before_homogeneous[i] << endl;
            //todo 齐次坐标往 projected平面投影
            corners_after_homogeneous[i] = (Homo_Projected[i] * corners_before_homogeneous[i].t()).t();
//            cout << corners_after_homogeneous[i] << endl;
        }
        //todo 每次投影过程中,选取最大拼接图片区间
        for (int i = 0; i < 4; ++i) {
            if(CornersInPano[0].x > corners_after_homogeneous[i].at<double>(0,0))
                CornersInPano[0].x = corners_after_homogeneous[i].at<double>(0,0);
            if(CornersInPano[0].y > corners_after_homogeneous[i].at<double>(0,1))
                CornersInPano[0].y = corners_after_homogeneous[i].at<double>(0,1);
            if(CornersInPano[1].x < corners_after_homogeneous[i].at<double>(0,0))
                CornersInPano[1].x = corners_after_homogeneous[i].at<double>(0,0);
            if(CornersInPano[1].y < corners_after_homogeneous[i].at<double>(0,1))
                CornersInPano[1].y = corners_after_homogeneous[i].at<double>(0,1);
        }

        Translation_all =(cv::Mat_<double>(3, 3) << 0,0,-CornersInPano[0].x,0,0,-CornersInPano[0].y,0,0,0);
//        cout << Translation_all << endl;
//    Trans_temp.copyTo(Translation_all);
        //panosize 是全景拼接图片的尺寸
        Size ProjectionSize = Size(
                CornersInPano[1].x - CornersInPano[0].x,
                CornersInPano[1].y - CornersInPano[0].y);
        PanoSize = ProjectionSize;
    }



};
void Projection::Homo_adjustment(){

}


Mat Projection::Pano_Projection(){

//    cout << "Single_ProjectMask" << endl;
    //todo 计算图片的整体
    Mat result,ProjMask;
    Rect dst_roi = resultRoi(CornersInPano, SinglePrjSize);
//    std::cout << "dst_roi size = " << dst_roi.size() << std::endl;

    result.create(dst_roi.size(), CV_8UC3);
    result.setTo(cv::Scalar::all(0));
    ProjMask.create(dst_roi.size(), CV_8UC3);
    ProjMask.setTo(cv::Scalar::all(0));
    //todo 把图片的的
    for(int i = 0; i < PicNums_; ++i)
    {
        Mat gray;
        cvtColor(Imgs_Projected[i], gray, COLOR_RGB2GRAY);

        int dx = CornersInPano[i].x - dst_roi.x;
        int dy = CornersInPano[i].y - dst_roi.y;
//        std::cout << "dx = " << dx << ", dy = " << dy << std::endl;
        Homo_adjustment();
        Imgs_Projected[i].copyTo(result(Rect(dx, dy, Imgs_Projected[i].cols, Imgs_Projected[i].rows)), gray);
    }
//    imwrite("./result/resultpano"+to_string(PicNums_)+".jpg",result);
    return result;
}
void Projection::Projection_forAll(){

}
void Projection::Store_the_Picture(){

}