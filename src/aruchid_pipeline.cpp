//
// Created by aruchid on 2019/12/7.
//

//#include "../include/aruchid_2pic.h"
//#include "../include/aruchid_log.h"
//#include "../include/aruchid_rotation.h"
//#include "../include/old/aruchid_matcher.h"
#include "../include/aruchid_printpic.h"
#include "../include/aruchid_pipeline.h"
#include "../include/aruchid_featuresfind.hpp"
#include "../include/aruchid_get_homo.h"
//#include "../include/aruchid_printpic.h"



Pipeline::Pipeline(){};

void Pipeline::Release() {

}

void Pipeline::Init(char ** argv){
    //todo this path must with "/"
    string inputpath  = argv[1];
    Path_datafolder_ = inputpath;
    Path_data_rotation_Info_ = inputpath +"photo_information.txt";
    //todo 设定相机内参
//    Kamera.focal = 800;
//    Kamera.aspect = 30;//todo 这里现在指的是Deepth
//    Mat imginput_ = imread(Path_datafolder_ +"1.jpg",IMREAD_COLOR);//要从1.jpg开始
//    resize(imginput_,imginput_,Size(imginput_.cols/4,imginput_.rows/4));
//    Kamera.ppx = imginput_.cols/2;
//    Kamera.ppy = imginput_.rows/2;
}

void Pipeline::GetDataFrame(){
    using namespace std;
    //todo reference
    // 依次：目标变换图片
    //      存储文件夹

    //todo 根据输入2确定文件夹位置
    vector<vector<double>> rotation_msg;//rotation info

    //todo 读取旋转信息并且按照顺序打印,同时读取所有图片到vector
//    Get_rotation_info(Path_data_rotation_Info_,rotation_msg);
    //图片张数
    int number_of_picture = rotation_msg.size();
    //单组图片堆
    //    vector <Mat> imginput;
//    cout << "本组共有 "<<rotation_msg.size() << " 张图片" <<endl;
    //显示每组图片各自的角度
    for (int i = 0; i < number_of_picture; ++i) {
        Mat imginput_ = imread(Path_datafolder_ + to_string(i+1)+".jpg",IMREAD_COLOR);//要从1.jpg开始
//        resize(imginput_,imginput_,Size(imginput_.cols/4,imginput_.rows/4));
//        cout << imginput_.size << endl;
        imgs.push_back(imginput_);
    }
//    cout << "读取图片张数 " << imgs.size() << endl << "各自的旋转信息如下：\n";
    for (auto PRY : rotation_msg){
        timestamp.push_back(PRY[1]);
        Eular_angle.push_back(Point3d(PRY[2],PRY[3],PRY[4]));
//        cout << PRY[0] << "\t" << PRY[1] << "\t" << PRY[2] << "\t"  << PRY[3] << "\t" << PRY[4] << "\t" << endl;
    }
    Img_Numbers = int(imgs.size());
//    cout << Eular_angle << endl;

}


int FindhBPOV_native(vector<Mat> Homos,Mat img0){
    int PanoSize_area;
    double app_start_time = getTickCount();
    int Img_Numbers = Homos.size();

    int PicNums_ = Img_Numbers;
    cout << "PICNUMS = " << PicNums_<< endl;
    int Best_POV_return = 0;
    for (int bestp = 0; bestp < Img_Numbers; ++bestp) {

        Mat tempEye = Mat::eye(3,3,CV_64F);
        vector<Mat> Homo_Projected;
        Projection Proj;
        bool Original = false;
        int BestPOV = bestp;
        for (int picnum = 0; picnum < PicNums_;) {
            if(picnum < BestPOV){
                Mat tempHomo;
                tempEye.copyTo(tempHomo);
                for (int step = picnum+1; step <= BestPOV ;) {
                    tempHomo = Homos[step].inv() * tempHomo;
                    step++;
                }
                Homo_Projected.push_back(tempHomo);
            }
            else if (picnum > BestPOV){
                Mat tempHomo;
                tempEye.copyTo(tempHomo);
                for (int step = BestPOV+1; step <= (picnum);) {
                    tempHomo =  tempHomo * Homos[step];
                    step++;
                }
                Homo_Projected.push_back(tempHomo);
            }
            else if (picnum == BestPOV){
                Mat tempHomo;//深拷贝问题
                tempEye.copyTo(tempHomo);
                Homo_Projected.push_back(tempHomo);
            }
            picnum++;

        }
        cout << "Homo_Projected" << Homo_Projected.size() << endl;
        Projection Prj;
        vector<Size> SinglePrjSize;//
        vector<Point> CornersInPano;
        vector<vector<Point2f>> corners(PicNums_);
        for (int i = 0; i < PicNums_; ++i) {
            vector<Point2f> tempcorners;
            Prj.CalcCorners(Homo_Projected[i], img0, corners[i]);
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
            CornersInPano.push_back(LT_position);//corners.left_top;
        }
        Rect dst_roi = resultRoi(CornersInPano, SinglePrjSize);
        cout << dst_roi.height <<" ";
        cout << dst_roi.width << endl;
        if (bestp==0){
            Best_POV_return = 0;
            PanoSize_area = dst_roi.height * dst_roi.width;
        }
        else{
            int PanoSize_area_tmp =  dst_roi.height*2 + dst_roi.width;
            if(PanoSize_area > PanoSize_area_tmp){
                PanoSize_area = PanoSize_area_tmp;
                Best_POV_return = bestp;
            }
        }
    }
    LOGLN("Best_POV Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    cout <<"Best_POV result: "<< Best_POV_return << endl;
    return Best_POV_return;
}

bool Pipeline::FindBestPOV(){
    int PanoSize_area;
    double app_start_time = getTickCount();
//    cout << "start find best pov" << endl;
    for (int bestp = 0; bestp < Img_Numbers; ++bestp) {

        Projection Proj;
        bool Original = false;
//    cout <<"imgs.size() "<< imgs.size() << endl;
//    cout <<"imgs.size() "<< imgs.size() << endl;
//        cout << imgs.size() << endl;
////    imwrite("./testprj.jpg",imgs[0]);
//        cout << Homo.size() << endl;
        int tempBest_POV = bestp;
//        cout << Best_POV << endl;
        Proj.InfoStack(imgs,Homo,tempBest_POV);

//        cout<< "Proj.PanoSize" << endl;
//        cout<< "Proj.Best_POV" << tempBest_POV << endl;
        Proj.PanoSize = Proj.Img_FieldScale_Find();
//        cout<< Proj.PanoSize << endl;
        if (bestp==0){
            Best_POV = 0;
            PanoSize_area = Proj.PanoSize.height*Proj.PanoSize.width;
        }
        else{
            int PanoSize_area_tmp =  Proj.PanoSize.height*2 + Proj.PanoSize.width;
            if(PanoSize_area>PanoSize_area_tmp){
                PanoSize_area = PanoSize_area_tmp;
                Best_POV = bestp;
            }
        }
    }
    LOGLN("Best_POV Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
//
    cout <<"Best_POV result: "<< Best_POV << endl;
    Best_POV = (int)(Img_Numbers/2);
    return 0;
//    vector<double> Angle_Sum;
    //找最小的角度合，权宜之机
//    for (int i = 0; i < Img_Numbers; ++i) {
//        Angle_Sum.push_back(Eular_angle[i].x + Eular_angle[i].y + Eular_angle[i].z);
//    }
//    float Best_POV_Sum = Angle_Sum[int(Img_Numbers/2)];
////    Best_POV = int(Img_Numbers/2);
//    Best_POV = 3;
//    int TempBPOV;
//    for (int i = 0; i < Img_Numbers; ++i) {
//        if(i>(Img_Numbers/2))
//            if  (Best_POV_Sum > Angle_Sum[i]){
//                Best_POV_Sum = Angle_Sum[i];
//                Best_POV = i ;
//            }
//    }

//    cout << "Best_POV = " << Best_POV << endl;

}

bool  Pipeline::GetHomoOfNew(Mat Newimg,Point3d NewEularAngle){


    vector<KeyPoint> keypoints_target_single, keypoints_trans_single;
    int Rotated_ = imgs.size()-1;
    int Target_ = Rotated_-1;
    Mat H_temp = FindMatch_CurseH(
            imgs,
            Rotated_,Target_,
            &keypoints_trans_single,
            &keypoints_target_single,
            Eular_angle,Best_POV
    );
    H_temp = Simplely_findHomo(
            keypoints_trans_single,
            keypoints_target_single
    );

//    H_temp = ECC_refineHomo(imgs[Target_],imgs[Rotated_],
//            keypoints_trans_single,
//            keypoints_target_single
//            );

//    cout << "transECC"<< endl;
    H_temp.at<double>(0,2) = 4*H_temp.at<double>(0,2);
    H_temp.at<double>(1,2) = 4*H_temp.at<double>(1,2);
    H_temp.at<double>(2,0) = 0.25*H_temp.at<double>(2,0);
    H_temp.at<double>(2,1) = 0.25*H_temp.at<double>(2,1);
    Mat temp = mywarpPerspective(imgs[Rotated_],H_temp);
//    imwrite("./result/result1210/"+to_string(1)+".jpg",temp);
//    cout << H_resize << endl;
    Homo.push_back(H_temp);
    //refresh
//    Rotated_ ++;
//    Target_ ++;
//    step++;
    keypoints_trans.clear();
    keypoints_target.clear();
    PointMatch.clear();
//    cout << "GetHOMO = " << Homo.size() << endl;

}


void Pipeline::GetHomoOfAll(){
    bool test_2pic = false;
//    int HomoSize_Set = imgs.size() - 1;
    int ImgPair_Size = imgs.size() -1;
    int Rotated_ = 1,Target_ = 0;

//    Rotation Rot;
    Mat H_temp,dst_pic,match_pic;
//    cout << "imgs.size" << imgs.size() << endl;

    // todo 两张图片测试
    //注意这里循环的个数为图片数-1
    if(ImgPair_Size!=0)
        Homo.push_back(Mat::eye(3,3,imgs[0].depth()));
    for (int step = 0; step < ImgPair_Size; )
    {
        H_temp = FindMatch_CurseH(
                imgs,
                Rotated_,Target_,
                &keypoints_trans,
                &keypoints_target,
                Eular_angle,Best_POV
        );
            H_temp = Simplely_findHomo(
                    keypoints_trans,
                    keypoints_target
            );
//        H_temp = ECC_refineHomo(
//                imgs[Target_],imgs[Rotated_],
//                keypoints_target,
//                keypoints_trans
//                );
//            cout << "H_temp2 " << H_temp << endl;
//        cout << Rotated_ << " " <<  Target_ << endl;
        Mat temp = mywarpPerspective(imgs[Rotated_],H_temp);
//        imwrite("./result/result1210/"+to_string(step+1)+".jpg",temp);
        Homo.push_back(H_temp);
        //refresh
        Rotated_ ++;
        Target_ ++;
        step++;
        keypoints_trans.clear();
        keypoints_target.clear();
        PointMatch.clear();
    }
//    cout << "GetHOMO = " << Homo.size() << endl;

}

void Pipeline::AllImgProjection(){

//    ofstream download_homo("./result/homos.txt");
//    for (int i = 0; i < Img_Numbers; ++i) {
//        for (int j = 0; j < 3; ++j) {
//            for (int k = 0; k < 3; ++k) {
//                download_homo << Homo[i].at<float>(j,k) <<",";
//            }
//        }
//        download_homo << endl;
//    }
//    cout << Homo[0].depth() << endl;
//    for (int i = 0; i < Img_Numbers; ++i) {
//        download_homo<< Homo[i] << endl;
//    }
//	cout << "into AllImgProjection" << endl;
    Projection Proj;
    bool Original = false;
//    cout <<"imgs.size() "<< imgs.size() << endl;
//    cout <<"imgs.size() "<< imgs.size() << endl;
//	cout << imgs.size() << endl;
//	imwrite("./testprj.jpg",imgs[0]);
//	cout << Homo.size() << endl;
//	cout << Best_POV << endl;
    Proj.InfoStack(imgs,Homo,Best_POV);
//	cout << "into InfoStack succ" << endl;
    Proj.Img_FieldTrans_Find();
//	cout << "into Img_FieldTrans_Find succ" << endl;
    Proj.Homo_adjustment();
//	cout << "into Homo_adjustment succ" << endl;
    PanoResult = Proj.Pano_Projection();
//	cout << "into Pano_Projection succ" << endl;
}


void Pipeline::SingleImgProjection(){

}


void Pipeline::ImgProjection(bool test)
{
    Mat TransAll;
    Size SizeOutput;
//    imgs

    Project_region_Get(imgs[Rotated],Homo[0],TransAll,SizeOutput);

//    cout << TransAll << endl;
//    cout << SizeOutput << endl;

    Mat me = cv::Mat::eye(cv::Size(3,3),TransAll.type());
//    cout << me << endl;
    Mat TransTarget = me+TransAll;
    warpPerspective(imgs[Rotated],dst_pic,Homo[0]+ TransAll,SizeOutput);
    //        drawMatches(imgs[Target],keypoints_target,imgs[Rotated],
    //                keypoints_trans,
    //                PointMatch,match_pic);
    Mat mask_ = Mat::zeros(imgs[Target].size(), imgs[Target].depth());
    mask_.setTo(255);
    Mat mask;
    warpPerspective(mask_,mask,TransTarget,SizeOutput);
//    imwrite("mask.jpg",mask);

    Mat dst_pic_target;
    warpPerspective(imgs[Target],dst_pic_target,TransTarget,SizeOutput);
//    imwrite("try_again1.jpg",dst_pic);
    dst_pic_target.copyTo(dst_pic,mask);
//    image02.copyTo(dst(Rect(0, 0, image02.cols, image02.rows)));
//    imwrite("try_again2.jpg",dst_pic);
//    imwrite("./")
}

