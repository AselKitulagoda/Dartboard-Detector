
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

Mat convolution2(cv::Mat &input, int kernel[3][3]){
Mat result(input.rows-2,input.cols-2,CV_32FC1,Scalar(0)); 
for (int j=1;j<=input.cols-1;j++){
    for (int i=1;i<=input.rows-1;i++){

        for (int x=3;x>0;x--){
            for (int y=3;y>0;y--){
                result.at<float>(i-1,j-1) += ((int) input.at<uchar>(i,j)*kernel[x][y]);
            }
            
        }
    }
}
return result;
}

int main( int argc, char** argv )
{
Mat img = imread("mandrill.jpg",1);
Mat grayimg(img.size(),CV_8UC1);
cvtColor(img,grayimg,CV_BGR2GRAY);
int kernel[3][3] = {{1,1,1},{1,1,1}};
Mat res = convolution2(grayimg,kernel)/9;
cv::normalize(res,res,0,255,cv::NORM_MINMAX);
imwrite("convolution.jpg",res);

waitKey(0);

res.release();

}
