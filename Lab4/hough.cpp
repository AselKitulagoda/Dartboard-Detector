// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace cv;

void sobel(
  cv::Mat &input,
  cv::Mat &output_mag,
  cv::Mat &output_dir);

Mat threshold(Mat input, int value)
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

  sobel(gray_image, image_mag, image_dir);

  Mat magnitude_img = imread( "mag.jpg", 0 );
  Mat direction_img = imread( "dir.jpg", 0 );  

  Mat thresholded = threshold(magnitude_img, 75);
  imwrite("thresholded.jpg", thresholded);

  return 0;
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

    Mat final_mag(output_mag.size(), CV_32FC1);
    Mat final_dir(output_dir.size(), CV_32FC1);
    cv::normalize(output_mag, final_mag, 0, 255, cv::NORM_MINMAX);
    cv::normalize(output_dir, final_dir, 0, 255, cv::NORM_MINMAX);
  imwrite( "mag.jpg", final_mag );
  imwrite( "dir.jpg", final_dir );
}