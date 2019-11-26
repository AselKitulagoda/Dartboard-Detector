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
#include "ellipse.hpp"
#include "line.hpp"

using namespace cv;
using namespace std;

/** Function Headers */
vector <Rect> detectAndDisplay1(Mat frame);

void hough_viola(std::vector <Rect> viodetected, std::vector <Rect> houghdetected,Mat frame);
 
/** Function to Calculate F1-Score */
void f1_score();

void draw_best_detected(cv::Mat original_img, 
            std::vector<Rect> hough, std::vector<Rect> ellipses);
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
    Mat lastoldframe = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat ellipseoldframe = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat allframe = imread(argv[1], CV_LOAD_IMAGE_COLOR);

 
    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_name)) { printf("--(!)Error loading cascade\n"); return -1; };
 
    // 3. Detect Faces and Display Result
    printf("before face det\n");
    vector <Rect> violaoutput = detectAndDisplay1(frame);
 
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
	
	// Hough Lines
	Mat hough_lines;
	// line_detection(thresholded_mag, unnormalised_dir, hough_lines);
	line_detection(thresholded_mag, unnormalised_dir, hough_lines);
	cv::imwrite("hough_lines.jpg", hough_lines);

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
    Mat thresholded_hough = thresholdd(new_hough_img, 150);

    cv::imwrite("thresholded_hough.jpg", thresholded_hough);

	printf("finished threshold hough\n");
    // Drawing the box around the detected stuff
    std::vector <Rect> hough_output;
	std::vector<Point2i> hough_centers;
    hough_output = draw_box(oldframe, hough_space, thresholded_hough, 115, hough_centers);

	for(int i = 0; i < hough_centers.size(); i++)
	{
		circle(allframe, hough_centers[i], 1, Scalar(0, 0, 255), 2);
	}
    // std::cout << "houghoutput in subtask3 x is : " << hough_output[0].x << std::endl;

    // cv::imwrite("rectangle.jpg", oldframe);

    // hough_viola(violaoutput,hough_output,lastoldframe);

    // cv::imwrite("violahough.jpg",lastoldframe);
	printf("starting ellipse\n");
    std::vector<RotatedRect> ellipses = ellipse_detector(magnitude_img, dir_img);
	printf("finished ellipse\n");
    std::vector<Rect> ellipse_output = convert_rotated_rect(ellipses);
	printf("converted ellipse\n");
    // draw_ellipses(ellipseoldframe,ellipses);
    // cv::imwrite("ellipse.jpg",ellipseoldframe);
    
    // draw_best_detected(allframe, hough_output, ellipse_output);
    cv::imwrite("all.jpg", allframe);
    return 0;
}
 
/** @function detectAndDisplay */
vector <Rect> detectAndDisplay1(Mat frame)
{
    std::vector<Rect> faces;
    std::vector<Rect> vout;
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
    return faces;
 
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

/** Function to get the detected rectangles*/
std::vector<Rect> detected_darts(int n) {
	std::vector<Rect> result;
	switch (n)
	{
	case 0: result.push_back(Rect(536, 288, 58, 58));
		result.push_back(Rect(146, 345, 52, 52));
		result.push_back(Rect(314, 82, 57, 57));
		result.push_back(Rect(578, 184, 57, 57));
		result.push_back(Rect(291, 202, 57, 57));
		result.push_back(Rect(516, 259, 62, 62));
		result.push_back(Rect(286, 84, 66, 66));
		result.push_back(Rect(306, 114, 69, 69));
		result.push_back(Rect(522, 216, 71, 71));
		result.push_back(Rect(334, 131, 76, 76));
		result.push_back(Rect(128, 186, 76, 76));
		result.push_back(Rect(455, 47, 134, 134));
		result.push_back(Rect(465, 10, 101, 101));
		result.push_back(Rect(394, 22, 179, 179));
		break;
	case 1: result.push_back(Rect(292, 313, 61, 61));
		result.push_back(Rect(286, 73, 63, 63));
		result.push_back(Rect(208, 138, 179, 179));
		result.push_back(Rect(264, 247, 135, 135));
		result.push_back(Rect(246, 59, 159, 159));
		break;
	case 2: result.push_back(Rect(241, 42, 54, 54));
		result.push_back(Rect(285, 46, 60, 60));
		result.push_back(Rect(321, 56, 53, 53));
		result.push_back(Rect(56, 172, 52, 52));
		result.push_back(Rect(144, 194, 54, 54));
		result.push_back(Rect(151, 247, 57, 57));
		result.push_back(Rect(117, 252, 52, 52));
		result.push_back(Rect(84, 79, 124, 124));
		result.push_back(Rect(45, 253, 65, 65));
		result.push_back(Rect(268, 95, 208, 208));
		result.push_back(Rect(404, 15, 197, 197));
		break;
	case 3: result.push_back(Rect(330, 158, 62, 62));
		result.push_back(Rect(202, 24, 94, 94));
		result.push_back(Rect(252, 74, 65, 65));
		result.push_back(Rect(374, 52, 84, 84));
		result.push_back(Rect(56, 70, 80, 80));
		result.push_back(Rect(298, 100, 135, 135));
		result.push_back(Rect(198, 8, 156, 156));
		break;
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
		result.push_back(Rect(82, 0, 333, 333)); 
		break;
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
	case 6: result.push_back(Rect(209, 103, 76, 76));
		result.push_back(Rect(154, 200, 52, 52));
		result.push_back(Rect(230, 330, 53, 53));
		result.push_back(Rect(189, 99, 67, 67));
		result.push_back(Rect(246, 161, 66, 66));
		result.push_back(Rect(114, 274, 63, 63));
		result.push_back(Rect(400, 28, 80, 80));
		result.push_back(Rect(352, 228, 92, 92));
		break;
	case 7: result.push_back(Rect(447, 95, 54, 54));
		result.push_back(Rect(451, 118, 56, 56));
		result.push_back(Rect(489, 262, 52, 52));
		result.push_back(Rect(619, 512, 53, 53));
		result.push_back(Rect(582, 498, 58, 58));
		result.push_back(Rect(633, 536, 57, 57));
		result.push_back(Rect(490, 229, 63, 63));
		result.push_back(Rect(146, 548, 70, 70));
		result.push_back(Rect(49, 559, 66, 66));
		result.push_back(Rect(193, 385, 69, 69));
		result.push_back(Rect(382, 343, 84, 84));
		result.push_back(Rect(176, 496, 96, 96));
		result.push_back(Rect(358, 474, 135, 135));
		result.push_back(Rect(224, 181, 146, 146));
		result.push_back(Rect(171, 151, 219, 219));
		break;
	case 8: result.push_back(Rect(256, 444, 53, 53));
		result.push_back(Rect(227, 447, 59, 59));
		result.push_back(Rect(498, 515, 55, 55));
		result.push_back(Rect(183, 638, 52, 52));
		result.push_back(Rect(266, 669, 52, 52));
		result.push_back(Rect(348, 684, 58, 58));
		result.push_back(Rect(240, 702, 52, 52));
		result.push_back(Rect(659, 203, 67, 67));
		result.push_back(Rect(863, 234, 82, 82));
		result.push_back(Rect(520, 540, 63, 63));
		result.push_back(Rect(275, 551, 62, 62));
		result.push_back(Rect(696, 594, 60, 60));
		result.push_back(Rect(209, 602, 61,61));
		result.push_back(Rect(232, 630, 60, 60));
		result.push_back(Rect(54, 229, 67, 67));
		result.push_back(Rect(62, 581, 63, 63));
		result.push_back(Rect(195, 680, 65, 65));
		result.push_back(Rect(566, 169, 69, 69));
		result.push_back(Rect(310, 585, 76, 76));
		result.push_back(Rect(326, 595, 76, 76));
		result.push_back(Rect(33, 255, 88, 88));
		result.push_back(Rect(30, 354, 80, 80));
		result.push_back(Rect(311, 394, 86, 86));
		result.push_back(Rect(245, 558, 88, 88));
		result.push_back(Rect(237, 672, 87, 87));
		result.push_back(Rect(37, 518, 102, 102));
		result.push_back(Rect(43, 194, 118, 118));
		result.push_back(Rect(799, 533, 142, 142));
		result.push_back(Rect(655, 118, 148, 148));
		result.push_back(Rect(782, 220, 176, 176));
		result.push_back(Rect(24, 204, 163, 163));
		result.push_back(Rect(284, 390, 171, 171));
		break;
	case 10: result.push_back(Rect(459, 54, 66, 66));
		result.push_back(Rect(905, 125, 62, 62));
		result.push_back(Rect(561, 307, 54, 54));
		result.push_back(Rect(134, 348, 52, 52));
		result.push_back(Rect(79, 355, 52, 52));
		result.push_back(Rect(557, 477, 58, 58));
		result.push_back(Rect(89, 516, 55, 55));
		result.push_back(Rect(48, 529, 52, 52));
		result.push_back(Rect(383, 564, 55, 55));
		result.push_back(Rect(949, 614, 52, 52));
		result.push_back(Rect(942, 637, 54, 54));
		result.push_back(Rect(834, 656, 61, 61));
		result.push_back(Rect(770, 658, 60, 60));
		result.push_back(Rect(556, 670, 52, 52));
		result.push_back(Rect(742, 666, 55, 55));
		result.push_back(Rect(571, 94, 81, 81));
		result.push_back(Rect(95, 543, 62, 62));
		result.push_back(Rect(585, 660, 61, 61));
		result.push_back(Rect(48, 215, 91, 91));
		result.push_back(Rect(196, 470, 66, 66));
		result.push_back(Rect(772, 639, 65, 65));
		result.push_back(Rect(122, 532, 69, 69));
		result.push_back(Rect(442, 67, 101, 101));
		result.push_back(Rect(573, 110, 102, 102));
		result.push_back(Rect(167, 515, 80, 80));
		result.push_back(Rect(703, 622, 76, 76));
		result.push_back(Rect(450, 496, 92, 92));
		result.push_back(Rect(59, 48, 158, 158));
		result.push_back(Rect(519, 134, 123, 123));
		result.push_back(Rect(22, 110, 149, 149));
		result.push_back(Rect(778, 56, 172, 172));
		result.push_back(Rect(386, 468, 180, 180));
		break;
	case 9: result.push_back(Rect(105, 369, 58, 58));
		result.push_back(Rect(92, 408, 54, 54));
		result.push_back(Rect(143, 258, 80, 80));
		result.push_back(Rect(120, 408, 93, 93));
		result.push_back(Rect(55, 190, 92, 92));
		result.push_back(Rect(140, 448, 96, 96));
		result.push_back(Rect(287, 475, 106, 106));
		result.push_back(Rect(236, 58, 204, 204));
		break;
	case 11: result.push_back(Rect(32, 8, 57, 57));
		result.push_back(Rect(350, 48, 74, 74));
		result.push_back(Rect(164, 83, 76, 76));
		result.push_back(Rect(278, 164, 72, 72));
		result.push_back(Rect(80, 22, 180, 180));
		break;
	case 12: result.push_back(Rect(376, 98, 54, 54));
		result.push_back(Rect(377, 127, 52, 52));
		result.push_back(Rect(84, 4, 63, 63));
		result.push_back(Rect(156, 204, 66, 66));
		result.push_back(Rect(127, 125, 106, 106));
		result.push_back(Rect(141, 96, 98, 98));
		result.push_back(Rect(115, 40, 174, 174));
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
	case 0: result.push_back(Rect(443, 14, (598 - 443), (193 - 14))); break; 
	case 1: result.push_back(Rect(197, 133, (391 - 197), (323 - 133))); break; 
	case 2: result.push_back(Rect(90, 83, (204 - 90), (198 - 83))); break; 
	case 3: result.push_back(Rect(325, 148, (388 - 325), (219 - 148))); break; 
	case 4: result.push_back(Rect(185, 97, (373 - 185), (291 - 97))); break;
	case 5: result.push_back(Rect(435, 140, (530 - 435), (246 - 140))); break;
	case 6: result.push_back(Rect(205, 108, (282 - 205), (189 - 108))); break; 
	case 7: result.push_back(Rect(255, 172, (389 - 255), (314 - 172))); break; 
	case 8: result.push_back(Rect(846, 218, (959 - 846), (337 - 218)));
			result.push_back(Rect(67, 251, (126 - 67), (340 - 251))); break;  
	case 9: result.push_back(Rect(204, 49, (436 - 204), (278 - 49))); break; 
	case 10: result.push_back(Rect(91, 103, (188 - 91), (214 - 103)));
			result.push_back(Rect(585, 129, (638 - 585), (212 - 129)));
			result.push_back(Rect(916, 152, (952 - 916), (212 - 152))); break; 
	case 11: result.push_back(Rect(168, 92, (241 - 168), (154 - 92))); break; 
	case 12: result.push_back(Rect(140, 93, (241 - 140), (209 - 93))); break; 
	case 13: result.push_back(Rect(287, 140, (402 - 287), (249 - 140))); break;
	case 14: result.push_back(Rect(121, 101, (247 - 121), (227 - 101)));
		result.push_back(Rect(989, 96, (1110 - 989), (218 - 96)));
		break;
	case 15: result.push_back(Rect(155, 57, (290 - 155), (199 - 57))); break;
	default: return result;
	}
	return result;
}

void hough_viola(std::vector <Rect> viodetected, std::vector <Rect> houghdetected,Mat frame){
    for (int i=0; i < viodetected.size(); ++i){
        for (int j = 0; j<houghdetected.size();++j){
            float _intersection = (viodetected[i] & houghdetected.front()).area();
			float _union = (viodetected[i] | houghdetected[j]).area();
            float iou = _intersection/_union;
            // std::cout << "houghdet_in hough_viola : " << houghdetected[j] << std::endl;
            // std::cout << "viodet in hough_viola : " << viodetected[i] << std::endl;
            std::cout << "iou : " << iou<< std::endl;
            if(iou > 0.6)
			{
                cv::rectangle(frame,houghdetected[j],Scalar(0,255,0),2);
			}


        }
    }
}

float calc_iou(Rect a, Rect b)
{
    float i = (a & b).area();
    float u = (a | b).area();
    return i/u;
}

// void draw_best_detected(cv::Mat original_img, 
//             std::vector<Rect> hough, std::vector<Rect> ellipses)
// {
//     std::vector<Rect> ground = ground_darts(9);
//     std::vector<Rect> viola = detected_darts(9);
//     std::cout << "GT size: " << ground.size() << std::endl;
//     for(int i = 0; i < ground.size(); i++)
//     {
//         // Get best Viola
//         float viola_iou = 0;
//         int viola_idx = 0;
//         for(int v = 0; v < viola.size(); v++)
//         {
//             if(calc_iou(ground[i], viola[v]) > viola_iou)
//             {
//                 viola_iou = calc_iou(ground[i], viola[v]);
//                 viola_idx = v;
//             }
//         }
// 		printf("did viola\n");
//         // Get best Hough
//         float hough_iou = 0;
//         int hough_idx = 0;
//         for(int h = 0; h < hough.size(); h++)
//         {
//             if(calc_iou(ground[i], hough[h]) > hough_iou)
//             {
//                 hough_iou = calc_iou(ground[i], hough[h]);
//                 hough_idx = h;
//             }
//         }
// 		printf("did hough\n");
//         // Get best Ellipse
//         float ellipse_iou = 0;
//         int ellipse_idx = 0;
//         for(int e = 0; e < ellipses.size(); e++)
//         {
//             if(calc_iou(ground[i], ellipses[e]) > ellipse_iou)
//             {
//                 ellipse_iou = calc_iou(ground[i], ellipses[e]);
//                 ellipse_idx = e;
//             }
//         }
// 		printf("did ellipse\n");

//         // Draw Everything
//         // Ground Truth = White
//         cv::rectangle(original_img, ground[i], Scalar(0, 0, 0), 2);
// 		printf("here1\n");
//         // Viola = Green
//         cv::rectangle(original_img, viola[viola_idx], Scalar(0, 255, 0), 2);
//         printf("here2\n");
//         // Hough = Red
//         cv::rectangle(original_img, hough[hough_idx], Scalar(255, 0, 0), 2);
// 		printf("here3\n");
//         // Ellipse = Blue
//         cv::rectangle(original_img, ellipses[ellipse_idx], Scalar(0, 0, 255), 2);
// 		printf("here4\n");
//     }
//     cv::imwrite("all.jpg", original_img);
// }