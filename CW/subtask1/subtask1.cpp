// dartboard.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		  
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Function to Calculate F1-Score */
double f1_score(double true_pos, double false_pos, double false_neg);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main(int argc, const char** argv)
{
	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay(frame);

	// 4. Save Result Image
	imwrite("detected.jpg", frame);

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

	// 4. Draw box around faces found
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
	}

}

double f1_score(double true_pos, double false_pos, double false_neg)
{
	std::cout << "True Positives: " << true_pos << std::endl;
	std::cout << "False Positives: " << false_pos << std::endl;
	std::cout << "False Negatives: " << false_neg << std::endl;

	double precision = true_pos / (true_pos + false_pos);
	double recall = true_pos / (true_pos + false_neg);

	if (true_pos == 0)
	{
		std::cout << 0 << std::endl;
		return 0;
	}

	double f1 = 2 * precision * recall / (precision + recall);
	std::cout << "F1-Score: " << f1 << std::endl;
	return f1;
}