#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;

//当前帧图片
Mat frame, gray,LastPic,src;
//前一帧图片
Mat prev_frame, prev_gray;
//保存特征点
vector<Point2f> features;
//初始化特征点
vector<Point2f> inPoints;

//当前帧和前一帧的点
vector<Point2f> fpts[2];
//特征点跟踪标志位
vector<uchar> status;
//误差和
vector<float> err;
//角点检测
void delectFeature(Mat &inFrame, Mat &ingray);
//画点
void drawFeature(Mat &inFrame);
//运动
void track();
//画运动轨迹
void drawLine();

int main()
{
    VideoCapture capture(0);
    // 摄像头读取文件开关
    if (capture.isOpened())
    {
        while (capture.read(frame)) {



            cvtColor(frame, gray, COLOR_BGR2GRAY);
            if (fpts[0].size() < 40) {
                delectFeature(frame, gray);
                fpts[0].insert(fpts[0].end(), features.begin(), features.end());
                inPoints.insert(inPoints.end(), features.begin(), features.end());

            }
            else {
                putText(frame, "follow", Point(100, 100), 1, 2, Scalar(255, 0, 0), 2, 8);
            }

            if (prev_gray.empty()) {
                gray.copyTo(prev_gray);
            }
            track();

            //保存当前帧为前一帧
            gray.copyTo(prev_gray);
            frame.copyTo(prev_frame);
            imshow("1", frame);
            waitKey(27);
        }
    }
    return 0;
}

void delectFeature(Mat &inFrame, Mat &ingray) {
    double maxCorners = 500.0;
    double qualityLevel = 0.01;
    double minDistance = 10.0;
    double blockSize = 3.0;
    double k = 0.04;
    goodFeaturesToTrack(ingray, features, maxCorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
    putText(frame, "get point", Point(100, 100), 1, 2, Scalar(255, 0, 0), 2, 8);

}

void drawFeature(Mat &inFrame) {
    for (size_t t = 0; t < fpts[0].size(); t++) {
        circle(inFrame, fpts[0][t], 2, Scalar(0, 255, 0), 2, 8);
    }
}

void track() {

    Point2d center = Point(gray.cols/2,gray.rows/2);
    calcOpticalFlowPyrLK(prev_gray, gray, fpts[0], fpts[1], status, err);
    int k = 0;
    for (int i = 0; i < fpts[1].size(); i++) {
        double dist = abs(fpts[0][i].x- fpts[1][i].x) + abs(fpts[0][i].y - fpts[1][i].y);
        if (dist > 2 && status[i]) {
            //删除损失的特征点
            inPoints[k] = inPoints[i];
            fpts[1][k++] = fpts[1][i];
        }
    }
    inPoints.resize(k);
    fpts[1].resize(k);
    drawLine();
    vector<Point2d> move;


    Point2d meanp_trans(0,0);
    if(k>50) {
        for (int i = 0; i < fpts[1].size(); ++i) {
            move.push_back(fpts[1][i] - inPoints[i]);
        }
        cout << move << endl;
        Mat mean_move_, std_movet;

        meanStdDev(move, mean_move_, std_movet);
        meanp_trans = Point2d(mean_move_.at<double>(0,0),mean_move_.at<double>(1,0));
//        cout << mean_move_ << endl;
        move.clear();
    }
    if (meanp_trans.x<5&&meanp_trans.y<5)
        cout << "stable" << endl;
    line(frame,center ,meanp_trans+center, Scalar(250, 0, 0), 5);
    swap(fpts[1], fpts[0]);
}

void drawLine() {
    for (size_t t = 0; t < fpts[1].size(); t++) {
        line(frame, inPoints[t], fpts[1][t], Scalar(0, 0, 255), 1, 8, 0);
        circle(frame, fpts[1][t], 2, Scalar(0, 255, 0), 2, 8, 0);
    }
}