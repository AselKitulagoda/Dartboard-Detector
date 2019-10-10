/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - RGBtoGRAY.cpp
//
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

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

 // CONVERT COLOUR AND SAVE
 Mat original;
//  cvtColor( image, gray_image, CV_BGR2GRAY );
bitwise_not(image,original);
 imwrite( "mandrill2convert.jpg", original );

 return 0;
}
