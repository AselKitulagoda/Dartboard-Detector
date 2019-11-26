#ifndef HOUGH_HPP
#define HOUGH_HPP

#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace cv;
using namespace std;

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	for (j = 0; j < dim2; j++) {
  	    array[i][j] = (int *) malloc(dim3 * sizeof(int));
	}

    }
    return array;
}

int ***create_hough_space(cv::Mat magnitude_img, cv::Mat direction_img, int min_radius, int max_radius, float rotation)
{
    int ***accumulator = malloc3dArray(magnitude_img.rows, magnitude_img.cols, max_radius);
    
    // Init accumulator to 0
    for(int y=0; y<magnitude_img.rows; y++)
    {
        for(int x=0; x<magnitude_img.cols; x++)
        {
            for(int r=0; r<max_radius; r++)
            {
                accumulator[y][x][r] = 0;
            }
        }
    }

    // Creating the Hough Space
    for(int y=0; y<magnitude_img.rows; y++)
    {
        for(int x=0; x<magnitude_img.cols; x++)
        {
            if(magnitude_img.at<uchar>(y, x) == 255)
            {
                for(int r = min_radius; r < max_radius; r++)
                {
                    for (float rotation=-M_PI;rotation <= M_PI;rotation+= M_PI/180)

                    {  
                    int x0, y0;

                    // Handling +
                    y0 = y + r*std::sin(direction_img.at<float>(y, x) + rotation);
                    x0 = x + r*std::cos(direction_img.at<float>(y, x) + rotation);

                    if(x0 >= 0 && y0 >= 0 && y0 < magnitude_img.rows && x0 < magnitude_img.cols)
                    {
                        accumulator[y0][x0][r] += 1;
                    }

                    // Handling -
                    y0 = y - r*std::sin(direction_img.at<float>(y, x) + rotation);
                    x0 = x - r*std::cos(direction_img.at<float>(y, x) + rotation);

                    if(x0 >= 0 && y0 >= 0 && y0 < magnitude_img.rows && x0 < magnitude_img.cols)
                    {
                        accumulator[y0][x0][r] += 1;
                    }
                }
                }
            }
        }
    }
    return accumulator;
}

Mat view_hough_space(int ***accumulator, cv::Mat magnitude_img, int min_radius, int max_radius)
{
    Mat hough(magnitude_img.rows, magnitude_img.cols, CV_32FC1, Scalar(0));
    for(int y=0; y<hough.rows; y++)
    {
        for(int x=0; x<hough.cols; x++)
        {
            for(int r=0; r<max_radius; r++)
            {
                hough.at<float>(y, x) += accumulator[y][x][r];
            }
        }
    }
    return hough;
}
#endif