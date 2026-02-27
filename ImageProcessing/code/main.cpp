#include "opencv2/opencv.hpp"
#include <iostream>
#include "string"
#include "string.h"
#include "vector"
#include "cmath"
#include "fstream"
#include "include/image_processing.h"
#include "include/experiment_run.h"

using namespace std;
using namespace cv;

int main()
{
	bool isDefaultSbj = false;
	bool isShowingImg = false;
	bool isSavingImg = false;
	bool isShowingRes = false;
	bool isSavingRes = true;

	int sbjIdx = 1;

	string extraSbjVideo = "data/raw_video/extra/sbj";
	string defSbjVideo = "data/raw_video/main/sbj";
	string videoPath;

	string defSavePath = "data/frames/main/";
	string extraSavePath = "data/frames/extra/";
	string savePath;

	if (isDefaultSbj)
	{
		videoPath = defSbjVideo;
		savePath = defSavePath;
	}
	else
	{
		videoPath = extraSbjVideo;
		savePath = extraSavePath;
	}

	//vector<string> depthLv = { "15cm", "20cm" , "25cm" , "30cm" , "35cm" , "40cm" , "45cm" , "50cm" , "55cm" , "60cm" };	
	//for(string depth: depthLv)
	//	saveFramesFromVideo(videoPath + to_string(sbjIdx) + "/" + depth + ".mp4", savePath, sbjIdx, 1);
	//

	vector<int> defaultSbjs = { 1, 5, 7, 10, 11 };
	vector<int> extraSbjs = { 2, 4, 8 };
	vector<int> sbjs;

	if (isDefaultSbj) sbjs = defaultSbjs;
	else sbjs  = extraSbjs;
	 
	for(vector<int>::iterator it = sbjs.begin(); it != sbjs.end(); it++)
		run(*it, isDefaultSbj, isShowingImg, isSavingImg, isShowingRes, isSavingRes);

	return 0;
}