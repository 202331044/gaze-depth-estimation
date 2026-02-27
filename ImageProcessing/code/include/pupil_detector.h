#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

class Pupil
{
public:
	Point2f center;
	float diameter;
	Rect boundingBox;
	Pupil(Point2f _center, float _diameter, Rect _boundingBox);
	Pupil();
};
