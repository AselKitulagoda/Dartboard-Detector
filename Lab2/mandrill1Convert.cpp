/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - thr.cpp
// TOPIC: RGB explicit thresholding
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() { 

  // Read image from file
  Mat image = imread("mandrill1.jpg", 1);
  Mat image2 = imread("mandrill1.jpg", 1);
//   Mat redImg;
//   // Threshold by looping through all pixels
//   for(int y=0; y<image.rows; y++) {
//    for(int x=0; x<image.cols; x++) {
//      image.at<Vec3b>(y,x)[0]=0;
//      image.at<Vec3b>(y,x)[1]=0;       
//        } 
//     }
//     redImg = image;
      for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
    //  image.at<Vec3b>(y,x)[2]=redImg.at<Vec3b>(y,x)[2];
     image2.at<Vec3b>((y+30)%image.cols, (x+30)%image.rows)[2]=image.at<Vec3b>(y,x)[2];       
       } 

    }
    imwrite("mandrill1convert.jpg",image2);





  //Save thresholded imagemage

  return 0;
}