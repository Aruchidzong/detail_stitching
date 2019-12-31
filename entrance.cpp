//
// Created by aruchid on 2019/12/11.
//

//
// Created by aruchid on 2019/12/2.
//

#include "./include/aruchid_pipeline.h"
#include "./include/aruchid_featuresfind.hpp"
#include "./stitch_connector.h"
#include "iostream"
#include <string>
//#include "./include/overlap.hpp"

vector<Point3d> Eular_angle_ex_;
vector<double> timestamp_ex_;
vector<Mat> Homo_ex_;
vector<Mat> imgs_ex_;


using namespace std;



bool new_Group = false;
//if(new_Group)


bool Stitch_Init() ;
bool StitchGroup_Finish();
bool Photo_Delete();
bool Photo_Increase(Mat Img_New,Point3d EularAngles_New,double timestamp_New,Mat &PanoResultOutput);

void Projectall_opticalflow(Mat Img_New,Point3d EularAngles_New,double timestamp_New,Mat Homo_New,Mat &PanoResultOutput);

bool Stitch_Init() {
	
	Pipeline Pipe;
//    Pipeline Pipe;
	Pipe.Img_Numbers = Pipe.imgs.size();
	cout << Pipe.Img_Numbers << Pipe.Img_Numbers << endl;
	Pipe.Best_POV = 0;
	
	Eular_angle_ex_.clear();
	timestamp_ex_.clear();
	Homo_ex_.clear();
	imgs_ex_.clear();
	cout << "Init successed"<< endl;
	
    return 1;
	
}

bool StitchGroup_Finish(){
	
    Pipeline Pipe;
	Pipe.imgs.clear();
	Pipe.Eular_angle.clear();
	Pipe.timestamp.clear();
	Pipe.Homo.clear();
	Pipe.Img_Numbers = Pipe.imgs.size();
	
	
	Eular_angle_ex_.clear();
	timestamp_ex_.clear();
	Homo_ex_.clear();
	imgs_ex_.clear();
}


bool Photo_Delete(){
	
    Pipeline Pipe;
//    Pipe.imgs.pop_back();
//    Pipe.Eular_angle.pop_back();
//    Pipe.timestamp.pop_back();
//    Pipe.Homo.pop_back();
	Pipe.Img_Numbers = Pipe.imgs.size();
	
	
	Eular_angle_ex_.pop_back();
	timestamp_ex_.pop_back();
	Homo_ex_.pop_back();
	imgs_ex_.pop_back();
    return 1;
	
	//todo 这里开始是findhomo的过程
}
bool Photo_Increase(Mat Img_New,Point3d EularAngles_New,double timestamp_New,Mat &PanoResultOutput){
	
//	cout << "Pipe Pipe successed"<< endl;
    Pipeline Pipe;
	
//	cout << "Pipe Pipe successed"<< endl;
    if(imgs_ex_.size() != 0)
    {
		Pipe.Eular_angle.swap(Eular_angle_ex_);
		Pipe.imgs.swap(imgs_ex_);
		Pipe.timestamp.swap(timestamp_ex_);
		Pipe.Homo.swap(Homo_ex_);
    }
	
//	cout << "Photo_Increase swap successed"<< endl;
	Pipe.Eular_angle.push_back(EularAngles_New);
	Pipe.imgs.push_back(Img_New);
	Pipe.timestamp.push_back(timestamp_New);
	Pipe.Img_Numbers = Pipe.imgs.size();
//    cout <<"Img_Numbers "<< Pipe.Img_Numbers <<endl;
//	imwrite("./imgspush.jpg",Pipe.imgs[0]);
//	cout << "Photo_Increase pushback successed"<< endl;
	//一张图一上再开始找
	if(Pipe.Img_Numbers >= 2)
		//如果找到的Homo New 不能用来拼接那就return 0,表示拼接失败
	{
		if(!Pipe.GetHomoOfNew(Img_New,EularAngles_New))
			return 0;
	}
	else{
    //  cout << "1111 "<< endl;
		Pipe.Homo.push_back(Mat::eye(3,3,Pipe.imgs[0].depth()));
	}
	
    //	cout << "Homo pushback successed"<< endl;
    //  cout << "homo size " <<Pipe.Homo.size() << endl;
    //  if(Pipe.Img_Numbers < 10)
    //	Pipe.FindBestPOV();
    Pipe.Best_POV = FindhBPOV_native(Pipe.Homo,Pipe.imgs[0]);
    //  Pipe.SingleImgProjection();
    //	cout << "Homo FindBestPOV successed"<< endl;
    //  else
	Pipe.AllImgProjection();
    //	cout << "Homo AllImgProjection successed"<< endl;
	
	
	Eular_angle_ex_.clear();
	imgs_ex_.clear();
	timestamp_ex_.clear();
	Homo_ex_.clear();
	
	Eular_angle_ex_.swap(Pipe.Eular_angle);
	imgs_ex_.swap(Pipe.imgs);
	timestamp_ex_.swap(Pipe.timestamp);
	Homo_ex_.swap(Pipe.Homo);
	
	Pipe.PanoResult.copyTo(PanoResultOutput);
	return 1;
	
}

void Projectall_opticalflow(Mat Img_New,Point3d EularAngles_New,double timestamp_New,Mat Homo_New,Mat &PanoResultOutput){
    //	cout << "Pipe Pipe successed"<< endl;
    Pipeline Pipe;

//	cout << "Pipe Pipe successed"<< endl;
    if(imgs_ex_.size() != 0)
    {
        Pipe.Eular_angle.swap(Eular_angle_ex_);
        Pipe.imgs.swap(imgs_ex_);
        Pipe.timestamp.swap(timestamp_ex_);
        Pipe.Homo.swap(Homo_ex_);
    }

//	cout << "Photo_Increase swap successed"<< endl;
    Pipe.Eular_angle.push_back(EularAngles_New);
    Pipe.imgs.push_back(Img_New);
    Pipe.timestamp.push_back(timestamp_New);
    Pipe.Img_Numbers = Pipe.imgs.size();
//    cout <<"Img_Numbers "<< Pipe.Img_Numbers <<endl;
//	imwrite("./imgspush.jpg",Pipe.imgs[0]);
//	cout << "Photo_Increase pushback successed"<< endl;
    //一张图一上再开始找
    if(Pipe.Img_Numbers >= 2)//如果找到的Homo New 不能用来拼接那就return 0,表示拼接失败
    {
        Pipe.Homo.push_back(Homo_New);
//            return 0;
    }
    else{
        //  cout << "1111 "<< endl;
        Pipe.Homo.push_back(Mat::eye(3,3,Pipe.imgs[0].depth()));
    }

    //	cout << "Homo pushback successed"<< endl;
    //  cout << "homo size " <<Pipe.Homo.size() << endl;
    //  if(Pipe.Img_Numbers < 10)
    //	Pipe.FindBestPOV();
    Pipe.Best_POV = FindhBPOV_native(Pipe.Homo,Pipe.imgs[0]);
    //  Pipe.SingleImgProjection();
    //	cout << "Homo FindBestPOV successed"<< endl;
    //  else
    Pipe.AllImgProjection();
    //	cout << "Homo AllImgProjection successed"<< endl;


    Eular_angle_ex_.clear();
    imgs_ex_.clear();
    timestamp_ex_.clear();
    Homo_ex_.clear();

    Eular_angle_ex_.swap(Pipe.Eular_angle);
    imgs_ex_.swap(Pipe.imgs);
    timestamp_ex_.swap(Pipe.timestamp);
    Homo_ex_.swap(Pipe.Homo);

    Pipe.PanoResult.copyTo(PanoResultOutput);
//    return 1;

};
void Realtime_Overlap_detect(Mat Fram_Img,vector<Point2d>OverLap_AreaCorners){

}







