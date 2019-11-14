// dartboard.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		  
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Function to get the detected rectangles*/
std::vector<Rect> detected_faces(int n);

/** Function to get the ground truth rectangles*/
std::vector<Rect> ground_faces(int n);

/** Function to Calculate F1-Score */
void f1_score();

/** Global variables */
String cascade_name = "/home/ks17226/Documents/ComputerVision/CW/frontalface.xml";
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

	f1_score();

	// 4. Save Result Image
	imwrite("detected.jpg", frame);

	return 0;
}

/** Function to get the detected rectangles*/
std::vector<Rect> detected_faces(int n) {
	std::vector<Rect> result;
	switch (n)
	{
	case 4: result.push_back(Rect(342, 107, 147, 147)); break;
	case 5: result.push_back(Rect(250, 164, 57, 57));
		result.push_back(Rect(290, 242, 63, 63));
		result.push_back(Rect(554, 244, 69, 69));
		result.push_back(Rect(673, 246, 64, 64));
		result.push_back(Rect(58, 249, 64, 64));
		result.push_back(Rect(513, 177, 55, 55));
		result.push_back(Rect(191, 214, 65, 65));
		result.push_back(Rect(641, 184, 59, 59));
		result.push_back(Rect(695, 599, 59, 59));
		result.push_back(Rect(425, 231, 68, 68));
		result.push_back(Rect(60, 135, 63, 63));
		result.push_back(Rect(377, 190, 57, 57));
		result.push_back(Rect(384, 400, 81, 81));
		result.push_back(Rect(528, 482, 170, 170));
		break;
	case 13: result.push_back(Rect(204, 164, 53, 53));
		result.push_back(Rect(412, 123, 122, 122));
		break;
	case 14: result.push_back(Rect(675, 219, 61, 61));
		result.push_back(Rect(394, 326, 67, 67));
		result.push_back(Rect(726, 190, 100, 100));
		result.push_back(Rect(577, 135, 87, 87));
		result.push_back(Rect(461, 216, 102, 102));
		result.push_back(Rect(558, 52, 315, 315));
		break;
	case 15: result.push_back(Rect(548, 121, 54, 54));
		result.push_back(Rect(65, 123, 86, 86));
		result.push_back(Rect(63, 366, 100, 100));
		result.push_back(Rect(124, 298, 147, 147));
		break;
	default: return result;
	}
	return result;
}

/** Function to get the ground truth rectangles*/
std::vector<Rect> ground_faces(int n) {
	std::vector<Rect> result;
	switch (n)
	{
	case 4: result.push_back(Rect(358, 141, 111, 101)); break;
	case 5: result.push_back(Rect(69, 142, (119 - 69), (193 - 142)));
		result.push_back(Rect(58, 260, (113 - 58), (310 - 260)));
		result.push_back(Rect(198, 229, (247 - 198), (274 - 229)));
		result.push_back(Rect(256, 174, (300 - 256), (216 - 174)));
		result.push_back(Rect(297, 254, (344 - 297), (299 - 254)));
		result.push_back(Rect(386, 196, (436 - 386), (239 - 196)));
		result.push_back(Rect(437, 247, (481 - 437), (292 - 247)));
		result.push_back(Rect(519, 187, (566 - 519), (229 - 187)));
		result.push_back(Rect(565, 258, (613 - 565), (306 - 258)));
		result.push_back(Rect(651, 195, (699 - 651), (238 - 195)));
		result.push_back(Rect(681, 256, (731 - 681), (306 - 256)));
		break;
	case 13: result.push_back(Rect(427, 146, (513 - 427), (232 - 146))); break;
	case 14: result.push_back(Rect(482, 238, (547 - 482), (305 - 482)));
		result.push_back(Rect(738, 210, (815 - 738), (283 - 210)));
		break;
	case 15: result.push_back(Rect(74, 141, (123 - 74), (208 - 141)));
		result.push_back(Rect(375, 115, (411 - 375), (184 - 115)));
		result.push_back(Rect(540, 138, (596 - 203), (203 - 138)));
		break;
	default: return result;
	}
	return result;
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
	for (int x = 0; x < faces.size(); x++)
	{
		std::cout << faces[x] << std::endl;
	}
	// 4. Draw box around faces found
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
	}

}

void f1_score()
{
	std::vector<Rect> detected = detected_faces(15);
	std::vector<Rect> ground = ground_faces(15);

	float actual_hits = 0;

	for (int d = 0; d < detected.size(); ++d)
	{
		for (int g = 0; g < ground.size(); ++g)
		{	
			float _intersection = (ground[g] & detected[d]).area();
			float _union = (ground[g] | detected[d]).area();

			float iou = _intersection/_union;
			if(iou > 0.4)
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
