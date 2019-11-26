#ifndef LINE_HPP
#define LINE_HPP

// example code:
// https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html

#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace cv;
using namespace std;

// https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
                      Point2f &r)
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

void line_detection(cv::Mat mag_img, cv::Mat dir_img, cv::Mat &hough_lines)
{
    // Init hough_lines image
    int hough_rows = (int) sqrt((mag_img.rows * mag_img.rows) + (mag_img.cols * mag_img.cols));
    int max_angle = 180;
    hough_lines.create(hough_rows, max_angle, mag_img.type());

    // Init Hough space
    int H[hough_rows][max_angle];
    for(int i = 0; i < hough_rows; i++)
    {
        for(int j = 0; j < max_angle; j++)
        {
            H[i][j] = 0;
        }
    }

    for(int y = 0; y < mag_img.rows; y++)
    {
        for(int x = 0; x < mag_img.cols; x++)
        {
            float theta = (dir_img.at<uchar>(y, x)/255) * 180;
            if(mag_img.at<uchar>(y, x) == 255)
            {
                float g = theta + 90;
                if(g > max_angle) g -= max_angle;

                float min_g, max_g;     // min and max gradient
                float tolerance = 5;    // margin of error

                min_g = g - tolerance;
                if(min_g < 0) min_g += max_angle;

                max_g = g + tolerance;
                if(max_g > max_angle) max_g -= max_angle;

                for(int t = 0; t < max_angle; t++)
                {
                    if(t >= min_g && t <= max_g)
                    {
                        float _theta = t * (CV_PI/180);
                        float rho = y*sin(_theta) + x*cos(_theta);

                        H[(int)rho][_theta] += 1;
                    }
                }
            }
        }
    }
    for(int i = 0; i < hough_rows; i++)
    {
        for(int j = 0; j < max_angle; j++)
        {
            if(H[j][i] > 255)
                H[j][i] = 255;
            hough_lines.at<uchar>(j, i) = H[j][i];
        }
    }
}

#endif