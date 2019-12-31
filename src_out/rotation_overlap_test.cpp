//
// Created by aruchid on 2019/12/13.
//

//

#include "../include/aruchid_pipeline.h"
#include "../include/aruchid_featuresfind.h"
#include "../include/aruchid_rotation.h"
#include "../stitch_connector.h"
#include "iostream"
#include <string>


void Get_rotation_info(string path,vector<vector<double>> &vv);

int main()
{

    string Path_data_rotation_Info_ = "./2019_12_05/16/photo_information.txt";
    vector<Mat> IMGS ;
    vector<double> timestamp_;
    vector<Point3d> Eular_angle_;

    vector<vector<double>> rotation_msg;//rotation info
    CameraParams Kamera;
    //todo 读取旋转信息并且按照顺序打印,同时读取所有图片到vector
    Get_rotation_info(Path_data_rotation_Info_,rotation_msg);
    cout << rotation_msg.size() << endl;


    for (int i = 0; i <rotation_msg.size()/2; ++i) {
        Mat tempreadimg = imread("./2019_12_05/16/"+to_string(i+1
        )+".jpg");
        IMGS.push_back(tempreadimg);
//		imwrite("./IMGS.jpg",IMGS[0]);
    }

    for (int i = 0; i <rotation_msg.size()/2; ++i){
        vector<double> PRY = rotation_msg[i];
        timestamp_.push_back(PRY[1]);
        Eular_angle_.push_back(Point3d(PRY[2],PRY[3],PRY[4]));
//        cout << PRY[0] << "\t" << PRY[1] << "\t" << PRY[2] << "\t"  << PRY[3] << "\t" << PRY[4] << "\t" << endl;
    }
//	cout << "Size" << endl;
//	cout << IMGS.size() << endl;
//	cout << Eular_angle_.size() << endl;
//	cout << timestamp_.size() << endl;

    Rotation Rt;
    double app_start_time = getTickCount();

    vector<Point> srcp;
    vector<Point> dsp;

//    Find_Overlap_area(IMGS[1],IMGS[0],Eular_angle_[1],?: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

    ImageFeatures feature_target;
    ImageFeatures feature_trans;
    vector<Point> srcpt(4);
    vector<Point> dstpt(4);

    int target,trans;
    target =1;
    trans  =0;
    set_src_feature(IMGS[target],feature_target);

    app_start_time = getTickCount();
    overlap_point(IMGS[trans],feature_target,Eular_angle_[trans],Eular_angle_[target],srcpt,dstpt);

    LOGLN("overlap_point , total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    line(IMGS[trans],dstpt[0],dstpt[1],Scalar(0,240,0),5);
    line(IMGS[trans],dstpt[1],dstpt[2],Scalar(0,240,0),5);
    line(IMGS[trans],dstpt[2],dstpt[3],Scalar(0,240,0),5);
    line(IMGS[trans],dstpt[3],dstpt[0],Scalar(0,240,0),5);
    imwrite("overlap_area.jpg",IMGS[trans]);
    cout << srcpt << endl;
    cout << dstpt << endl;
//    cout << H_overlap << endl;
    //	waitKey();
    //
    //	Photo_Increase(IMGS[1],Eular_angle_[1],timestamp_[1],Pano);
    //
    //	imwrite("pano2.jpg",Pano);
    ////	waitKey();
    //
    //	Photo_Increase(IMGS[2],Eular_angle_[2],timestamp_[3],Pano);
    //
    //	imwrite("pano3.jpg",Pano);
    //	waitKey();
}

void Find_Rot_Overlap(Mat Template,Mat Trans,Point3d Eular_Template,Point3d Eular_Trans){


}


void Get_rotation_info(string path, vector<vector<double>> &vv){
    cout << "-----------Get_rotation_info----------- " << endl;
    ifstream in(path);
    string line;
    while (getline(in, line)){
        stringstream ss(line);
        string tmp;
        vector<double> v;
        while (getline(ss, tmp, ',')){//按“，”隔开字符串
            v.push_back(stod(tmp));//stod: string->double
        }
        vv.push_back(v);
    }
}