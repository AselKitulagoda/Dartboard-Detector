#ifndef DETECTOR_HPP
#define DETECTOR_HPP

using namespace std;
using namespace cv;

#define MIN_VOTES 15

void draw_box(Mat original_img, int ***accumulator, Mat thresholded_hough, int max_radius)
{
    std::vector<Point3i> xyr_vals;
    for(int y = 0; y < thresholded_hough.rows; y++)
    {
        for(int x = 0; x < thresholded_hough.cols; x++)
        {
            if(thresholded_hough.at<uchar>(y, x) == 255)
            {
                int argmax = 0, max = 0;
                for(int r = max_radius; r >= 0; r--)
                {
                    if(accumulator[y][x][r] > max_radius)
                    {
                        max = accumulator[y][x][r];
                        argmax = r;
                    }
                }
                if(accumulator[y][x][argmax] > MIN_VOTES)
                {
                    xyr_vals.push_back(Point3i(y, x, argmax));
                    break;
                }
            }
        }
    }

    // Drawing the rectangles from the xyr_vals
    for(auto val : xyr_vals)
    {
        int r = val.z;
        Point p1 = Point(val.x - r, val.y - r);
        Point p2 = Point(val.x + r, val.y + r);

        cv::rectangle(original_img, p1, p2, Scalar(0, 255, 0), 2);
    }
}
#endif