#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace cv;
using namespace std;

Mat thresholdd(Mat input, int value)
{
    Mat output = Mat(input.rows, input.cols, CV_8UC1, Scalar(0));
    for(int y=0; y<input.rows; y++)
    {
        for(int x=0; x<input.cols; x++)
        {
            if(input.at<uchar>(y, x) > value)
            {
                output.at<uchar>(y, x)  = 255;
            }
            else
            {
                output.at<uchar>(y, x) = 0;
            }
        }
    }
    return output;
}

Mat float_thresholdd(Mat input, int value)
{
    Mat output = Mat(input.rows, input.cols, CV_8UC1, Scalar(0));
    for(int y=0; y<input.rows; y++)
    {
        for(int x=0; x<input.cols; x++)
        {
            if(input.at<float>(y, x) > value)
            {
                output.at<uchar>(y, x)  = 255;
            }
            else
            {
                output.at<uchar>(y, x) = 0;
            }
        }
    }
    return output;
}

Mat normalise(Mat img)
{
    Mat normalised;
    cv::normalize(img, normalised, 0, 255, cv::NORM_MINMAX);
    return normalised;
}

void sobel(cv::Mat &input, cv::Mat &output_mag, cv::Mat &output_dir)
{
  Mat dx_kernel = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

  Mat dy_kernel = dx_kernel.t();

    for(int i=1; i<input.rows - 1; i++)
    {
        for(int j=1; j<input.cols - 1; j++)
        {
            float pixelX = 0;
            float pixelY = 0;
            for(int x=-1; x<=1; x++)
            {
                for(int y=-1; y<=1; y++)
                {
                    pixelX += input.at<uchar>(i-x, j-y) * dx_kernel.at<int>(x+1, y+1);
                    pixelY += input.at<uchar>(i-x, j-y) * dy_kernel.at<int>(x+1, y+1);
                }
            }
            output_mag.at<float>(i, j) = sqrt(pixelX*pixelX + pixelY*pixelY);
            output_dir.at<float>(i, j) = atan2(pixelY, pixelX);
        }
    }
}
#endif