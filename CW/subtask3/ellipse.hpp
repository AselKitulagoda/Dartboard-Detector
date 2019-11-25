#ifndef ELLIPSE_HPP
#define ELLIPSE_HPP

#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>
#include "sobel.hpp"

using namespace std;
using namespace cv;

#define PI 3.1415

std::vector<RotatedRect> ellipse_detector(Mat mag_img, Mat dir_img) 
{
    // Edge detection on magnitude img
    Mat thresholded_mag = thresholdd(mag_img, 70);

    // vector of vectors for contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // Finding the contours
    findContours(thresholded_mag, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // Vector to store the ellipses
    vector<RotatedRect> ellipses(contours.size());

    for(int i = 0; i < contours.size(); i++)
    {
        if(contours[i].size() > 100)
        {
            ellipses[i] = fitEllipse(Mat(contours[i]));
        }
    }
    return ellipses;
}

std::vector<Rect> convert_rotated_rect(std::vector<RotatedRect> ellipses)
{
    // https://stackoverflow.com/questions/32920419/opencv-minimum-upright-bounding-rect-of-a-rotatedrect
    std::vector<Rect> bounding_rectangles;

    for(int i = 0; i < ellipses.size(); i++)
    {
        float degree = ellipses[i].angle * PI/180;
        float majorAxe = ellipses[i].size.width / 2;
        float minorAxe = ellipses[i].size.height / 2;
        float x = ellipses[i].center.x;
        float y = ellipses[i].center.y;
        float c_degree = cos(degree);
        float s_degree = sin(degree);
        float t1 = atan(-(majorAxe * s_degree) / (minorAxe * c_degree));
        float c_t1 = cos(t1);
        float s_t1 = sin(t1);
        float w1 = majorAxe * c_t1 * c_degree;
        float w2 = minorAxe * s_t1 * s_degree;
        float maxX = x + w1-w2;
        float minX = x - w1+w2;

        t1 = atan((minorAxe * c_degree) / (majorAxe * s_degree));
        c_t1 = cos(t1);
        s_t1 = sin(t1);
        w1 = minorAxe * s_t1 * c_degree;
        w2 = majorAxe * c_t1 * s_degree;
        float maxY = y + w1+w2;
        float minY = y - w1-w2;
        if (minY > maxY)
        {
            float temp = minY;
            minY = maxY;
            maxY = temp;
        }
        if (minX > maxX)
        {
            float temp = minX;
            minX = maxX;
            maxX = temp;
        }
        Rect rect(minX, minY, maxX-minX+1, maxY-minY+1);
        bounding_rectangles.push_back(rect);
    }
    return bounding_rectangles;
}

void draw_ellipses(Mat original_img, std::vector<RotatedRect> ellipses)
{
    for(int i = 0; i < ellipses.size(); i++)
    {
        cv::ellipse(original_img, ellipses[i], Scalar(0, 0, 255), 2, 8);
    }
}

#endif