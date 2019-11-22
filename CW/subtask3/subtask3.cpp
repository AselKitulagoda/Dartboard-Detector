// header inclusion
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include "detector.hpp"
#include "hough.hpp"
#include "sobel.hpp"


using namespace cv;
using namespace std;



/** Function Headers */
void detectAndDisplay(Mat frame);
 
/** Function to Calculate F1-Score */
void f1_score();
 
/** Global variables */
// String cascade_name = "frontalface.xml";
String cascade_name = "cascade.xml";
CascadeClassifier cascade;
 
/** @function main */
int main(int argc, const char** argv)
{
    // 1. Read Input Image
    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat oldframe = imread(argv[1], CV_LOAD_IMAGE_COLOR);
 
    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_name)) { printf("--(!)Error loading cascade\n"); return -1; };
 
    // 3. Detect Faces and Display Result
    printf("before face det\n");
    detectAndDisplay(frame);
 
    // f1_score();
 
    // 4. Save Result Image
    imwrite("cascade_detected.jpg", frame);
 
    // Hough Transform stuff starts here
    Mat gray_image;
    cvtColor(oldframe, gray_image, CV_BGR2GRAY);

 
    Mat mag_img(gray_image.rows, gray_image.cols, CV_32FC1, Scalar(0));
    Mat dir_img(gray_image.rows, gray_image.cols, CV_32FC1, Scalar(0));
    printf("here\n");
 
    // Performing Sobel Edge detection, getting the magnitude and direction imgs
    sobel(gray_image, mag_img, dir_img);
    printf("after sobel\n");
    cv::imwrite("magnew.jpg", mag_img);
    cv::imwrite("dirnew.jpg", dir_img);
    Mat magnitude_img = imread( "magnew.jpg", 0 );

 
    // Storing the unnormalised direction
    Mat unnormalised_dir = dir_img;
 
    // Thresholding the magnitude
    Mat thresholded_mag = thresholdd(normalise(magnitude_img), 70);
    cv::imwrite("thresholded_mag.jpg", thresholded_mag);
 
    // Creating the hough space, assuming 0 rotation.
    // Min radius and Max radius 40 and 115 respectively
    int ***hough_space = create_hough_space(thresholded_mag, unnormalised_dir, 40, 115, 0);
 
    // Generating the hough image
    Mat hough_img = view_hough_space(hough_space, thresholded_mag, 40, 115);
    Mat final_hough(mag_img.rows, mag_img.cols, CV_8UC1, Scalar(0));
 
    cv::normalize(hough_img, final_hough, 0, 255, NORM_MINMAX);
    cv::imwrite("normalised_hough_img.jpg", final_hough);
 
    // Reading in normalised hough image to ensure values are uchar instead of float
    Mat new_hough_img = imread("normalised_hough_img.jpg", 0);
 
    // Thresholding the newly read hough image
    Mat thresholded_hough = thresholdd(new_hough_img, 180);

    cv::imwrite("thresholded_hough.jpg", thresholded_hough);

 
    // Drawing the box around the detected stuff
    draw_box(oldframe, hough_space, thresholded_hough, 115);

    cv::imwrite("rectangle.jpg", oldframe);

 
    return 0;
}
 
/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
 
    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
 
    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
 
    // 3. Print number of Faces found
    std::cout << faces.size() << std::endl;
 
    // Printing all the detected rectangles
    // for (int x = 0; x < faces.size(); x++)
    // {
    //  std::cout << faces[x] << std::endl;
    // }
 
    // 4. Draw box around faces found
    printf("draw boxes before\n");
    for (int i = 0; i < faces.size(); i++)
    {
        rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
        std::cout << faces[i].height << std::endl;

        
    }
    printf("draw boxes after\n");
 
}
 
void f1_score()
{
    // std::vector<Rect> detected = detected_darts(15);
    // std::vector<Rect> ground = ground_darts(15);
    std::vector<Rect> detected;
    std::vector<Rect> ground;
    float actual_hits = 0;
 
    for (int d = 0; d < detected.size(); ++d)
    {
        for (int g = 0; g < ground.size(); ++g)
        {  
            float _intersection = (ground[g] & detected[d]).area();
            float _union = (ground[g] | detected[d]).area();
 
            float iou = _intersection/_union;
            if(iou > 0.59)
            {
                actual_hits += 1;
            }
            std::cout << "IOU: " << iou << std::endl;
        }
    }
    float tpr = actual_hits / ground.size();
    float fpr = 1 - tpr;
    float fnr = actual_hits - ground.size();
    float precision = actual_hits / detected.size();
    float f1 = 2 * tpr * precision / (precision + tpr);
 
    std::cout << "Actual Faces: " << ground.size() << std::endl;
    std::cout << "Detected Faces: " << detected.size() << std::endl;
    std::cout << "Actual Hits: " << actual_hits << std::endl;
    std::cout << "TPR: " << tpr << std::endl;
    std::cout << "FPR: " << fpr << std::endl;
    std::cout << "FNR: " << fnr << std::endl;
    std::cout << "F1-Score: " << f1 << std::endl;
}