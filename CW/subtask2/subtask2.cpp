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
std::vector<Rect> detected_darts(int n);

/** Function to get the ground truth rectangles*/
std::vector<Rect> ground_darts(int n);

/** Function to Calculate F1-Score */
void f1_score();

/** Global variables */
// String cascade_name = "frontalface.xml";
String cascade_name = "negatives/dartcascade/cascade.xml";
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
	imwrite("cascade_detected/dart15_detected.jpg", frame);

	return 0;
}

/** Function to get the detected rectangles*/
std::vector<Rect> detected_darts(int n) {
	std::vector<Rect> result;
	switch (n)
	{
	case 4: result.push_back(Rect(557, 205, 54, 54));
		result.push_back(Rect(498, 328, 54, 54));
		result.push_back(Rect(484, 357, 54, 54));
		result.push_back(Rect(306, 466, 52, 52));
		result.push_back(Rect(156, 604, 63, 63));
		result.push_back(Rect(551, 70, 65, 65));
		result.push_back(Rect(93, 236, 65, 65));
		result.push_back(Rect(516, 373, 71, 71));
		result.push_back(Rect(84, 275, 124, 124));
		result.push_back(Rect(404, 26, 138, 138));
		result.push_back(Rect(180, 114, 184, 184));
		result.push_back(Rect(342, 328, 142, 142));
		result.push_back(Rect(146, 26, 160, 160));
		result.push_back(Rect(229, 10, 189, 189));
		result.push_back(Rect(82, 0, 333, 333)); break;
	case 5: result.push_back(Rect(357, 44, 52, 52));
		result.push_back(Rect(291, 50, 54, 54));
		result.push_back(Rect(534, 90, 52, 52));
		result.push_back(Rect(261, 355, 52, 52));
		result.push_back(Rect(375, 364, 54, 54));
		result.push_back(Rect(252, 56, 61, 61));
		result.push_back(Rect(332, 92, 63, 63));
		result.push_back(Rect(67, 242, 60, 60));
		result.push_back(Rect(354, 277, 57, 57));
		result.push_back(Rect(386, 167, 101, 101));
		result.push_back(Rect(425, 134, 130, 130));
		result.push_back(Rect(634, 376, 142, 142));
		break;
	case 13: result.push_back(Rect(385, 131, 59, 59));
		result.push_back(Rect(344, 174, 70, 70));
		result.push_back(Rect(242, 197, 65, 65));
		result.push_back(Rect(306, 136, 104, 104));
		result.push_back(Rect(30, 60, 109, 109));
		result.push_back(Rect(30, 118, 101, 101));
		result.push_back(Rect(156, 120, 131, 131));
		result.push_back(Rect(18, 21, 149, 149));
		result.push_back(Rect(293, 148, 149, 149));
		result.push_back(Rect(247, 13, 271, 271));
		break;
	case 14: result.push_back(Rect(320, 62, 54, 54));
		result.push_back(Rect(323, 128, 54, 54));
		result.push_back(Rect(982, 69, 149, 149));
		result.push_back(Rect(1171, 202, 52, 52));
		result.push_back(Rect(660, 302, 54, 54));
		result.push_back(Rect(205, 334, 55, 55));
		result.push_back(Rect(354, 350, 83, 83));
		result.push_back(Rect(390, 460, 55, 55));
		result.push_back(Rect(1158, 464, 58, 58));
		result.push_back(Rect(1130, 536, 54, 54));
		result.push_back(Rect(33, 563, 57, 57));
		result.push_back(Rect(948, 102, 63, 63));
		result.push_back(Rect(995, 323, 60, 60));
		result.push_back(Rect(1056, 325, 63, 63));
		result.push_back(Rect(1001, 444, 60, 60));
		result.push_back(Rect(1130, 448, 57, 57));
		result.push_back(Rect(9, 456, 62, 62));
		result.push_back(Rect(42, 472, 62, 62));
		result.push_back(Rect(706, 506, 60, 60));
		result.push_back(Rect(323, 192, 64, 64));
		result.push_back(Rect(1155, 276, 63, 63));
		result.push_back(Rect(136, 511, 70, 70));
		result.push_back(Rect(532, 45, 71, 71));
		result.push_back(Rect(946, 141, 72, 72));
		result.push_back(Rect(28, 517, 72, 72));
		result.push_back(Rect(1124, 537, 72, 72));
		result.push_back(Rect(1025, 165, 96, 96));
		result.push_back(Rect(1116, 504, 99, 99));
		result.push_back(Rect(512, 113, 131, 131));
		result.push_back(Rect(529, 199, 122, 122));
		result.push_back(Rect(326, 316, 138, 138));
		result.push_back(Rect(455, 365, 128, 128));
		result.push_back(Rect(414, 75, 155, 155));
		result.push_back(Rect(91, 49, 233, 233));
		result.push_back(Rect(464, 26, 207, 207));
		result.push_back(Rect(599, 38, 263, 263));
		result.push_back(Rect(104, 317, 281, 281));
		break;
	case 15: result.push_back(Rect(36, 219, 69, 69));
		result.push_back(Rect(186, 55, 87, 87));
		result.push_back(Rect(155, 56, 173, 173));
		break;
	default: return result;
	}
	return result;
}

/** Function to get the ground truth rectangles*/
std::vector<Rect> ground_darts(int n) {
	std::vector<Rect> result;
	switch (n)
	{
	case 4: result.push_back(Rect(152, 60, (377 - 152), (306 - 60))); break;
	case 5: result.push_back(Rect(418, 124, (549 - 418), (258 - 124))); break;
	case 13: result.push_back(Rect(254, 100, (421 - 254), (276 - 100))); break;
	case 14: result.push_back(Rect(104, 81, (264 - 204), (248 - 82)));
		result.push_back(Rect(970, 74, (1129 - 970), (237 - 74)));
		break;
	case 15: result.push_back(Rect(130, 33, (304 - 130), (216 - 33))); break;
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
	// for (int x = 0; x < faces.size(); x++)
	// {
	// 	std::cout << faces[x] << std::endl;
	// }
	// 4. Draw box around faces found
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
	}

}

void f1_score()
{
	std::vector<Rect> detected = detected_darts(15);
	std::vector<Rect> ground = ground_darts(15);

	float actual_hits = 0;

	for (int d = 0; d < detected.size(); ++d)
	{
		for (int g = 0; g < ground.size(); ++g)
		{	
			float _intersection = (ground[g] & detected[d]).area();
			float _union = (ground[g] | detected[d]).area();

			float iou = _intersection/_union;
			if(iou > 0.5)
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
