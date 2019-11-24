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

#define MIN_VOTES 4

using namespace cv;
using namespace std;

std::vector<Point2f> line_detector(cv::Mat original_img, cv::Mat mag_img, int threshold)
{
    std::vector<Vec2f> lines;
    std::vector<Point2f> line_coords;

    Mat canny_img;
    Canny(mag_img, canny_img, 50, 200, 3);

    HoughLines(mag_img, lines, 1, CV_PI/180, threshold, 0, 0);

    int H[mag_img.rows][mag_img.cols];
    for(int y = 0; y < mag_img.rows; y++)
    {
        for(int x = 0; x < mag_img.cols; x++)
        {
            H[y][x] = 0;
        }
    }

    for(int i = 0; i < lines.size(); i++)
    {
        for(int j = i+1; j < lines.size(); j++)
        {
            float rho1 = lines[i][0];
            float theta1 = lines[i][1];
            Point pt11, pt12;
            double a1 = cos(theta1);
            double b1 = sin(theta1);
            double x10 = a1 * rho1;
            double y10 = b1 * rho1;

            pt11.x = cvRound(x10 + 1000*(-b1));
            pt11.y = cvRound(y10 + 1000*(a1));
            pt12.x = cvRound(x10 - 1000*(-b1));
            pt12.y = cvRound(y10 - 1000*(a1));

            float rho2 = lines[j][0];
            float theta2 = lines[j][1];
            Point pt21, pt22;
            double a2 = cos(theta2);
            double b2 = sin(theta2);
            double x20 = a2 * rho2;
            double y20 = b2 * rho2;

            pt21.x = cvRound(x20 + 1000*(-b2));
            pt21.y = cvRound(y20 + 1000*(a2));
            pt22.x = cvRound(x20 - 1000*(-b2));
            pt22.y = cvRound(y20 - 1000*(a2));

            Point2f r;

            if(intersection(pt11, pt12, pt21, pt22, r))
            {
                if(r.y < canny_img.rows && r.x < canny_img.cols
                && r.y > 0 && r.x > 0)
                {
                    H[r.y][r.x] += 1;
                }
            }
        }
    }
    for(int y = 0; y < canny_img.rows; y++)
    {
        for(int x = 0; x < canny_img.cols; x++)
        {
            if(H[y][x] == 8)
            {
                line_coords.push_back(Point2f(y, x));
                cv::circle(original_img, Point2f(y, x), 7, Scalar(0, 0, 255), -1);
            }
        }
    }
    return line_coords;
}

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

#endif