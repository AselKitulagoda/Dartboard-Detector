// header inclusion
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

#define min_radius 20
#define max_radius 100

using namespace cv;

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

Mat sobel(
  cv::Mat &input,
  cv::Mat &output_mag,
  cv::Mat &output_dir);

Mat hough_transform(
    cv::Mat magnitude_img,
    cv::Mat direction_img
);

Mat thresholdd(Mat input, int value)
{
    Mat output = Mat(input.rows, input.cols, CV_8UC1, Scalar(0));
    for(int i=0; i<input.rows; i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            if(input.at<uchar>(i, j) > value)
            {
                output.at<uchar>(i, j)  = 255;
            }
            else
            {
                output.at<uchar>(i, j) = 0;
            }
        }
    }
    return output;
}

Mat hough_transform(cv::Mat magnitude_img, cv::Mat direction_img)
{
    int center_x = magnitude_img.rows;
    int center_y = magnitude_img.cols;
    int ***accumulator = malloc3dArray(center_x, center_y, max_radius);

    for(int x=0; x<center_x; x++)
    {
        for(int y=0; y<center_y; y++)
        {
            for(int z=0; z<max_radius; z++)
            {
                accumulator[x][y][z] = 0;
            }
        }
    }

    for(int x=0; x<magnitude_img.rows; x++)
    {
        for(int y=0; y<magnitude_img.cols; y++)
        {
            if(magnitude_img.at<uchar>(x, y) == 255)
            {
                for(int r = min_radius; r < max_radius; r++)
                {   
                    int x0, y0;

                    // Handling +
                    x0 = x + r*std::sin(direction_img.at<float>(x, y));
                    y0 = y + r*std::cos(direction_img.at<float>(x, y));

                    if(x0 >= 0 && y0 >= 0 && x0 < magnitude_img.rows && y0 < magnitude_img.cols)
                    {
                        accumulator[x0][y0][r] += 1;
                    }

                    // Handling -
                    x0 = x - r*std::sin(direction_img.at<float>(x, y));
                    y0 = y - r*std::cos(direction_img.at<float>(x, y));

                    if(x0 >= 0 && y0 >= 0 && x0 < magnitude_img.rows && y0 < magnitude_img.cols)
                    {
                        accumulator[x0][y0][r] += 1;
                    }
                }
            }
        }
    }

    Mat hough(magnitude_img.rows, magnitude_img.cols, CV_32FC1, Scalar(0));
    for(int x=0; x<magnitude_img.rows; x++)
    {
        for(int y=0; y<magnitude_img.cols; y++)
        {
            for(int r=0; r<max_radius; r++)
            {
                hough.at<float>(x, y) += accumulator[x][y][r];
            }
        }
    }


    Mat final_hough(magnitude_img.rows, magnitude_img.cols, CV_8UC1, Scalar(0));
    cv::normalize(hough, final_hough, 0, 255, NORM_MINMAX);

    return final_hough;
}

int main( int argc, char** argv )
{
  // LOADING THE IMAGE
  char* imageName = argv[1];

  Mat image;
  image = imread( imageName, 1 );

  if( argc != 2 || !image.data )
  {
    printf( " No image data \n " );
    return -1;
  }

  // CONVERT COLOUR, BLUR AND SAVE
  Mat gray_image;
  cvtColor( image, gray_image, CV_BGR2GRAY );

  Mat image_mag(gray_image.rows, gray_image.cols, CV_32FC1, Scalar(0));
  Mat image_dir(gray_image.rows, gray_image.cols, CV_32FC1, Scalar(0));

  Mat unnormalised_dir = sobel(gray_image, image_mag, image_dir);

  Mat magnitude_img = imread( "mag.jpg", 0 );
//   Mat direction_img = imread( "dir.jpg", 0 );  

  Mat thresholded = thresholdd(magnitude_img, 70);
  cv::imwrite("thresholded.jpg", thresholded);
  
  Mat hough = hough_transform(thresholded, unnormalised_dir);
  cv::imwrite("hough.jpg", hough);

  Mat new_hough = imread("hough.jpg", 0);

  Mat thresholded_hough = thresholdd(new_hough, 70);
  cv::imwrite("thresholded_hough.jpg", thresholded_hough);
  
//   Mat recognised(thresholded)

  return 0;
}

Mat sobel(cv::Mat &input, cv::Mat &output_mag, cv::Mat &output_dir)
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

    Mat final_mag(output_mag.size(), CV_32FC1);
    Mat final_dir(output_dir.size(), CV_32FC1);
    cv::normalize(output_mag, final_mag, 0, 255, cv::NORM_MINMAX);
    cv::normalize(output_dir, final_dir, 0, 255, cv::NORM_MINMAX);
  cv::imwrite( "mag.jpg", final_mag );
  cv::imwrite( "dir.jpg", final_dir );
  return output_dir;
}