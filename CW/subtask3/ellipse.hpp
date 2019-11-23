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

std::vector<RotatedRect> ellipse_detector(Mat mag_img, Mat dir_img) 
{
    // Edge detection on magnitude img
    Mat thresholded_mag = thresholdd(mag_img, 70);

    // vector of vectors for contours
    vector<vector<Point>> contours;
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

void draw_ellipses(Mat original_img, std::vector<RotatedRect> ellipses)
{
    for(int i = 0; i < ellipses.size(); i++)
    {
        cv::ellipse(original_img, ellipses[i], Scalar(0, 0, 255), 2, 8);
    }
}

#endif