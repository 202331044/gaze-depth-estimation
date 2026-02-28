#include "pupil_detector.h"

Pupil::Pupil()
{

};

Pupil::Pupil(cv::Point2f _center, float _diameter, cv::Rect _boundingBox) : center(_center), diameter(_diameter), boundingBox(_boundingBox)
{
};