//
// Created by aruchid on 2019/12/13.
//

//

#include "./include/aruchid_pipeline.h"
#include "./include/aruchid_featuresfind.hpp"
#include "./stitch_connector.h"
#include "iostream"
#include <string>


void Get_rotation_info(string path,vector<vector<double>> &vv);

int main()
{

string Path_data_rotation_Info_ = "./2019_12_05/15/photo_information.txt";
	vector<Mat> IMGS ;
	vector<double> timestamp_;
	vector<Point3d> Eular_angle_;
	
//	vector<vector<double>> rotation_msg;//rotation info
//
//	//todo 读取旋转信息并且按照顺序打印,同时读取所有图片到vector
//	Get_rotation_info(Path_data_rotation_Info_,rotation_msg);
//	cout << rotation_msg.size() << endl;
//
//
//	for (int i = 0; i <rotation_msg.size()/2; ++i) {
//		Mat tempreadimg = imread("./2019_12_05/15/"+to_string(i+1
//				)+".jpg");
//		IMGS.push_back(tempreadimg);
////		imwrite("./IMGS.jpg",IMGS[0]);
//	}
//
//	for (int i = 0; i <rotation_msg.size()/2; ++i){
//	    vector<double> PRY = rotation_msg[i];
//		timestamp_.push_back(PRY[1]);
//		Eular_angle_.push_back(Point3d(PRY[2],PRY[3],PRY[4]));
////        cout << PRY[0] << "\t" << PRY[1] << "\t" << PRY[2] << "\t"  << PRY[3] << "\t" << PRY[4] << "\t" << endl;
//	}
//	cout << "Size" << endl;
//	cout << IMGS.size() << endl;
//	cout << Eular_angle_.size() << endl;
//	cout << timestamp_.size() << endl;




	Mat Pano;
//	Stitch_Init();

    Point3d Eular_Fake(0,0,0);
    for (int step = 0; step < IMGS.size(); ++step) {
        Photo_Increase(IMGS[step],Eular_Fake,0,Pano);
        imwrite("pano"+to_string(step)+".jpg",Pano);
    }

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