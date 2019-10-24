// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);

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

 Mat carBlurred;
 GaussianBlur(gray_image,23,carBlurred);

 imwrite( "filter2d.jpg", carBlurred );

 return 0;
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

       // SET KERNEL VALUES
	for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
	  for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
           kernel.at<double>(m+ kernelRadiusX, n+ kernelRadiusY) = (double) 1.0/(size*size);
			
       }

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 1; i < input.rows - 1; i++ )
	{	
		for( int j = 1; j < input.cols - 1; j++ )
		{	
			int pixel_values[9] = {(int) input.at<uchar>(i - 1, j - 1),
										  (int) input.at<uchar>(i - 1, j),
										  (int) input.at<uchar>(i - 1, j + 1),
										  (int) input.at<uchar>(i, j - 1),
										  (int) input.at<uchar>(i, j),
										  (int) input.at<uchar>(i, j + 1),
										  (int) input.at<uchar>(i + 1, j - 1),
										  (int) input.at<uchar>(i + 1, j),
										  (int) input.at<uchar>(i + 1, j + 1)};

			int n = sizeof(pixel_values)/sizeof(pixel_values[0]); 
			std::sort(pixel_values, pixel_values + n);

			int median_index = 9/2;
			int median = pixel_values[median_index];

			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) median;
		}
	}
}