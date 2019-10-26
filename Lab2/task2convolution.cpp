#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main()
{
    Mat image = imread("mandrill.jpg", 0);
    Mat result = Mat(image.rows, image.rows, CV_8UC1);
    Mat kernel = Mat(3, 3, CV_8UC1);

    // Initialising the Kernel Matrix (3x3)
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            kernel.at<uchar>(i, j) = 1;
        }
    }

    // 5x5
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            kernel.at<uchar>(i, j) = 1;
        }
    }

    // Convoluting the Image
    for(int i=1; i<image.rows - 1; i++)
    {
        for(int j=1; j<image.cols - 1; j++)
        {
            int pixel = 0;
            for(int x=-1; x<=1; x++)
            {
                for(int y=-1; y<=1; y++)
                {
                    pixel += image.at<uchar>(i-x, j-y) * kernel.at<uchar>(x+1, y+1);
                }
            }
            result.at<uchar>(i, j) = pixel/9;
        }
    }
    imwrite("convolution.jpg", result);

    return 0;
}