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

int **create2DArray(int width, int height)
{
    int **H = (int **) malloc(sizeof(int *)*width);
    for(int i = 0; i < width; i++)
    {
        H[i] = (int *) malloc(height * sizeof(int));
    }
    return H;
}

Mat line_detection(cv::Mat mag_img, cv::Mat dir_img)
{   
    // Init hough_lines image
    int hough_rows = (int) sqrt((mag_img.rows * mag_img.rows) + (mag_img.cols * mag_img.cols));
    int max_angle = 360;

    Mat test(hough_rows, max_angle, CV_32FC1, Scalar(0));

    // Init Hough space
    int **H = create2DArray(hough_rows, max_angle);
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
            float theta = dir_img.at<float>(y, x);
            if(mag_img.at<uchar>(y, x) == 255)
            {   
                int rho = x*cos(theta) + y*sin(theta);
                if(rho < hough_rows && rho >= 0)
                {   
                    int deg = (theta * 180/M_PI) + (max_angle/2);
                    H[rho][deg] += 1;
                }
            }
        }
    }

    for(int x = 0; x < hough_rows; x++)
    {
        for(int y = 0; y < max_angle; y++)
        {   
            test.at<float>(x, y) = H[x][y];
        }
    }
    cv::normalize(test, test, 0, 255, NORM_MINMAX);
    threshold(test, test, 60, 255, THRESH_BINARY);

    Mat straight_lines(mag_img.rows, mag_img.cols, CV_32FC1, Scalar(0));
    Mat flattened(mag_img.rows, mag_img.cols, CV_8UC1, Scalar(0));
    for(int i = 0; i < hough_rows; i++)
    {
        for(int j = 0; j < max_angle; j++)
        {   
            if(test.at<float>(i, j) == 255)
            {
                for(int x = 0; x < mag_img.cols; x++)
                {   
                    float theta = (j - (max_angle/2)) * M_PI/180;
                    int y = (i - x*cos(theta))/sin(theta);
                    if(y >= 0 && y < mag_img.rows)
                    {
                        straight_lines.at<float>(y, x) += 1;
                    }            
                }
            }
        }
    }
    cv::normalize(straight_lines, flattened, 0, 255, NORM_MINMAX);
    return flattened;
}

std::vector<Point> get_intersection_points(cv::Mat flattened)
{
    std::vector<Point> points;
    // threshold(flattened, flattened, 140, 255, THRESH_BINARY);
    flattened = float_thresholdd(flattened, 140);
    cv::imwrite("thr.jpg", flattened);
    for(int i = 0; i < flattened.rows; i++)
    {
        for(int j = 0; j < flattened.cols; j++)
        {
            if(flattened.at<uchar>(i, j) != 0)
            {
                points.push_back(Point(j, i));
            }
        }
    }
    return points;
}

#endif