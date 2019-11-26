#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace std;
using namespace cv;

#define MIN_VOTES 150

std::vector<Rect> draw_box(Mat original_img, int ***accumulator, Mat thresholded_hough, int max_radius, std::vector<Point> &hough_centers)
{
    vector <Rect> houghoutput;
    std::vector<int> votes;
    std::vector<Point3i> xyr_vals;
    for(int y = 0; y < thresholded_hough.rows; y++)
    {
        for(int x = 0; x < thresholded_hough.cols; x++)
        {
            if(thresholded_hough.at<uchar>(y, x) == 255)
            {
                int argmax = 0, max = 0;
                for(int r = max_radius; r >= 0; r--)
                {
                    if(accumulator[y][x][r] > max)
                    {
                        max = accumulator[y][x][r];
                        argmax = r;
                    }
                }
                if(accumulator[y][x][argmax] > MIN_VOTES)
                {   
                    votes.push_back(accumulator[y][x][argmax]);
                    xyr_vals.push_back(Point3i(y, x, argmax));
                    // break;
                }
            }
        }
    }

    // Drawing the rectangles from the xyr_vals
    for(int i = 0; i < xyr_vals.size(); i++)
    {   
        int r = xyr_vals[i].z;
        Point p1 = Point(xyr_vals[i].y - r, xyr_vals[i].x - r);
        Point p2 = Point(xyr_vals[i].y + r, xyr_vals[i].x + r);
        hough_centers.push_back(Point(xyr_vals[i].y, xyr_vals[i].x));
    //    cv::rectangle(originautoal_img, p1, p2, Scalar(0, 255, 0), 2);
        houghoutput.push_back(Rect(p1.x, p1.y, abs(p2.x - p1.x), abs(p2.y - p1.y)));
    //    std::cout <<     //    std::cout << "houghoutput x in detector is : " << houghoutput[0].x << std::endl;
// "houghoutput x in detector is : " << houghoutput[0].x << std::endl;
    }
    groupRectangles(houghoutput, votes, 2, 0.5);
    return houghoutput;
}
#endif