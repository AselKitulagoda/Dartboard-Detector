// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

#define PI 3.14159265

using namespace cv;

void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);

void sobel(
  cv::Mat &input,
  cv::Mat &output_mag,
  cv::Mat &output_dir);

Mat threshold(
    cv::Mat input,
    float value
);

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

  Mat image_mag(gray_image.size(), CV_32FC1);
  Mat image_dir(gray_image.size(), CV_32FC1);

  sobel(gray_image, image_mag, image_dir);

  return 0;
}

Mat threshold(Mat input, float value)
{
    Mat output = Mat(input.rows, input.cols, CV_8UC1);
    for(int i=0; i<input.rows; i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            if(input.at<float>(i, j) > value)
            {
                output.at<float>(i, j)  = 255;
            }
            else
            {
                output.at<float>(i, j) = 0;
            }
        }
    }
    return output;
}

void sobel(cv::Mat &input, cv::Mat &output_mag, cv::Mat &output_dir)
{
  Mat dx_kernel = cv::Mat(3, 3, CV_32FC1);
  dx_kernel.at<float>(0, 0) = -1;
  dx_kernel.at<float>(0, 1) = 0;
  dx_kernel.at<float>(0, 2) = -1;
  dx_kernel.at<float>(1, 0) = -2;
  dx_kernel.at<float>(1, 1) = 0;
  dx_kernel.at<float>(1, 2) = 2;
  dx_kernel.at<float>(2, 0) = -1;
  dx_kernel.at<float>(2, 1) = 0;
  dx_kernel.at<float>(2, 2) = 1;

  Mat dy_kernel = dx_kernel.t();

  // we need to create a padded version of the input
  // or there will be border effects
  int kernelRadiusX = ( dx_kernel.size[0] - 1 ) / 2;
  int kernelRadiusY = ( dx_kernel.size[1] - 1 ) / 2;

  cv::Mat paddedInput;
  cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

    // now we can do the convolution
    for(int i=0; i<input.rows; i++)
    {
      for(int j=0; j<input.cols; j++)
      {
        float sum_x = 0.0;
        float sum_y = 0.0;

        for(int m=-kernelRadiusX; m<=kernelRadiusX; m++)
        {
          for(int n=-kernelRadiusY; n<=kernelRadiusY; n++)
          {
            // find the correct indices we are using
            int imagex = i + m + kernelRadiusX;
            int imagey = j + n + kernelRadiusY;
            int kernelx = m + kernelRadiusX;
            int kernely = n + kernelRadiusY;

            // get the values from the padded image and the kernel
            int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
            float kernelvalX = dx_kernel.at<float>( kernelx, kernely );
            float kernelvalY = dy_kernel.at<float>( kernelx, kernely );

            // do the multiplication
            sum_x += imageval * kernelvalX;
            sum_y += imageval * kernelvalY;
          }
        }
      // set the output value as the sum of the convolution
	  output_mag.at<float>(i, j) = sqrt(pow(sum_x, 2) + pow(sum_y, 2));
	  output_dir.at<float>(i, j) = atan((sum_x)/(sum_y)) * 180/PI;
      }
    }
    cv::normalize(output_mag, output_mag, 0, 255, cv::NORM_MINMAX);
    // cv::normalize(output_dir, output_dir, 0, 255, cv::NORM_MINMAX);
  imwrite( "mag.jpg", output_mag );
  imwrite( "dir.jpg", output_dir );
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);
	
	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convolution
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		
		}
	}

}