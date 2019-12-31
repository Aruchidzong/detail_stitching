//
// Created by aruchid on 2019/12/19.
//

// 从摄像头中读取
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2//highgui/highgui.hpp>
#include <iostream>
#include "./include/overlap.hpp"


cv::detail::ImageFeatures feature_target;
cv::detail::ImageFeatures feature_trans;
std::vector<cv::Point> srcpt(4);
std::vector<cv::Point> dstpt(4);
cv::Mat src;
Point center;

vector<KeyPoint> srcKeypt;
Mat srcDescptor;

#define LOGLN(msg) std::cout << msg << std::endl
//int target,trans;

int main() {
    cv::namedWindow("暴风影音", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture cap;
    // 读取摄像头
    cap.open(0);
    // 判断摄像头是否打开
    if (!cap.isOpened()) {
        std::cerr << "Could't open capture" << std::endl;
        return -1;
    }
    cv::Mat frame;

    Mat Overlap_Area;
    // 接收键盘上的输入
    char keyCode;
    bool flag = false;


    // 保存的图片名称
    std::string imgName = "123.jpg";
    while (1) {
//        list<Point2d> move_extern_temp(8,Point2d(0,0));
//        for(auto mv:move_extern_temp){
//            cout << mv;
//        }
        cout << endl;
        // 把读取的摄像头传入Mat对象中
        cap >> frame;
        // 判断是否成功
        if (frame.empty()) {
            break;
        }
        // 把每一帧图片表示出来
        // 在30毫秒内等待是否存在键盘输入
        keyCode = cv::waitKey(30);
        if (keyCode == 's') {
            // 把图片保存起来
            cv::imwrite(imgName,frame);
            //  imgName.at(0)++;
            //  frame.release();
            cout << "s get";
            src = cv::imread("./123.jpg");
//            flag = set_src_feature(src,srcKeypt,srcDescptor);
            flag = set_src_feature(src,feature_trans);
            center  = Point(src.cols/2,src.rows/2);

            src.copyTo(Overlap_Area);
            Overlap_Area = Scalar::all(0);
        }
        Mat dst_,src_;
        if(flag){

            //        cout << "keycode"<< endl;

            cv::Point3d Eularangle(0,0,0);
            cv::Point3d Eularangle2(0,0,0);

            double t_start_time = getTickCount();
//            int color =  overlap_point(frame,srcKeypt,srcDescptor,Eularangle,Eularangle,srcpt,dstpt);
            int color =  overlap_point(frame,feature_trans,Eularangle,Eularangle2,srcpt,dstpt);
//            bool color =  overlap_point(frame,feature_trans,srcpt,dstpt);
//        trans(frame,src,srcpt,dstpt);
            LOGLN("overlap_point , total time: " << ((getTickCount() - t_start_time) / getTickFrequency()) << " sec\n");
//        cout<< dstpt[0] << endl;l
            if(color==1) {
//                cout << srcpt[2]/100-center << endl;
                line(frame, center, (srcpt[2]/10+center), Scalar(100, 100, 0), 5);

                frame.copyTo(src_);
                src_.copyTo(dst_);
                cv::Point pt[1][4];
                pt[0][0] = dstpt[3];
                pt[0][1] = dstpt[0];
                pt[0][2] = dstpt[1];
                pt[0][3] = dstpt[2];
//                pt[0][4] = cv::Point(450,350);
                const cv::Point* ppt[1]={pt[0]};
                int npt[1] = {4};
                cv::fillPoly(src_,ppt,npt,1,cv::Scalar(0,255,0));

                //cv::rectangle(src,cv::Point(450,100),cv::Point(750,400),cv::Scalar(0,255,0),-1,8);
                cv::addWeighted(dst_,0.7,src_,0.3,0,dst_);


//                line(frame, dstpt[0], dstpt[1], Scalar(0, 240, 0), 5);
//                line(frame, dstpt[1], dstpt[2], Scalar(0, 240, 0), 5);
//                line(frame, dstpt[2], dstpt[3], Scalar(0, 240, 0), 5);
//                line(frame, dstpt[3], dstpt[0], Scalar(0, 240, 0), 5);
//
//                circle(frame, srcpt[0], 5, Scalar(0, 240, 204), 10);
//                circle(frame, srcpt[1], 5, Scalar(0, 0, 70), 10);
//                fillPoly(Overlap_Area,dstpt,Scalar::all(200));
//                imshow("Overlap",Overlap_Area);
//        imwrite("overlap_area.jpg",frame);
                cv::imshow("暴风影音", dst_);
            }



        }

//        cout << srcpt << endl;
//        cout << dstpt << endl;


    }
    return 0;
}