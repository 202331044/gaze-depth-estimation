#pragma once
#include <opencv2/core.hpp>

class Pupil
{
public:
	cv::Point2f center;
	float diameter;
	cv::Rect boundingBox;
	Pupil(cv::Point2f _center, float _diameter, cv::Rect _boundingBox);
	Pupil();
};