//
// Created by aruchid on 2019/12/23.
//


#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void video2image(string video, string path)
{
    VideoCapture capture(video);
    long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT);
    cout << "total frames is:" << totalFrameNumber << "." << endl;
    //设置开始帧
    long frameToStart = 1;
    capture.set(CAP_PROP_POS_FRAMES, frameToStart);
    cout << "from" << frameToStart << "read" << endl;
    //设置结束帧
    int frameToStop = 500;

    //获取帧率
    double rate = capture.get(CAP_PROP_FPS);
    cout << "rate is:" << rate << endl;
    double delay = 1000 / rate;
    //定义一个用来控制读取视频循环结束的变量
    bool stop = false;
    long currentFrame = frameToStart;

    if (!capture.isOpened())
    {
        cerr << "Failed to open a video" << endl;
        return;
    }

    Mat frame;
    int num = 1;
    string filename;
    char   temp_file[15];

    while (!stop)
    {
        capture >> frame;
        if (frame.empty())
            break;
        string temp_file = to_string(num); //4表示字符长度,10表示十进制,_itoa_s实现整型转字符串
//        filename = temp_file;
        filename = path + temp_file + ".jpg";

        cout << "now is reading" << currentFrame << "." << endl;
        imshow("Extractedframe", frame);

        cout << "now is writing" << currentFrame << "." << endl;
        imwrite(filename, frame);

        int c = waitKey(delay);
        //按下ESC或者到达指定的结束帧后退出读取视频
        if ((char)c == 27 || currentFrame > frameToStop)
        {
            stop = true;
        }
        //按下按键后会停留在当前帧，等待下一次按键
        if (c >= 0)
        {
            waitKey(0);
        }

        num++;
        currentFrame++;
    }
    capture.release();
    waitKey(0);
}

int main(int argc, char** argv)
{
    string videoFromfile = "./VID_20191223_172716.mp4";  //读取视频
    string Imagespath = "./image/";    // 保存图片的文件夹路径一定要有，因为OpenCV不会自动创建文件夹
    video2image(videoFromfile, Imagespath);
    return 0;
}