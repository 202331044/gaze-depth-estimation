#include "opencv2/opencv.hpp"
#include <iostream>
#include "../include/pupil_detector.h"

using namespace std;
using namespace cv;

Pupil::Pupil()
{

};

Pupil::Pupil(Point2f _center, float _diameter, Rect _boundingBox)
{
	center = _center;
	diameter = _diameter;
	boundingBox = _boundingBox;
};