// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace cv;

void sobel(
  cv::Mat &input,
  cv::Mat &output_dx,
  cv::Mat &output_dy,
  cv::Mat &output_mag,
  cv::Mat &output_dir);

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
  
  Mat image_dx(gray_image.rows, gray_image.cols, CV_32FC1, Scalar(0));
  Mat image_dy(gray_image.rows, gray_image.cols, CV_32FC1, Scalar(0));
  Mat image_mag(gray_image.rows, gray_image.cols, CV_32FC1, Scalar(0));
  Mat image_dir(gray_image.rows, gray_image.cols, CV_32FC1, Scalar(0));

  sobel(gray_image, image_dx, image_dy, image_mag, image_dir);

  return 0;
}

void sobel(cv::Mat &input, cv::Mat &output_dx, cv::Mat &output_dy, cv::Mat &output_mag, cv::Mat &output_dir)
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
            output_dx.at<float>(i, j) = pixelX;
            output_dy.at<float>(i, j) = pixelY;
            output_mag.at<float>(i, j) = sqrt(pixelX*pixelX + pixelY*pixelY);
            output_dir.at<float>(i, j) = atan2(pixelY, pixelX);
        }
    }

    Mat final_dx(output_dx.size(), CV_32FC1);
    Mat final_dy(output_dy.size(), CV_32FC1);
    Mat final_mag(output_mag.size(), CV_32FC1);
    Mat final_dir(output_dir.size(), CV_32FC1);
    cv::normalize(output_dx, final_dx, 0, 255, cv::NORM_MINMAX);
    cv::normalize(output_dy, final_dy, 0, 255, cv::NORM_MINMAX);
    cv::normalize(output_mag, final_mag, 0, 255, cv::NORM_MINMAX);
    cv::normalize(output_dir, final_dir, 0, 255, cv::NORM_MINMAX);
  imwrite( "dx.jpg", final_dx );
  imwrite( "dy.jpg", final_dy );
  imwrite( "mag.jpg", final_mag );
  imwrite( "dir.jpg", final_dir );
}