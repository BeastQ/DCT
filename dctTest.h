#pragma once
#pragma once
#include <vector>;
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <iostream> 
#include<vector>
#include<string>;
using namespace cv;
using namespace std;
int m, n, t;
int msgLen;
int bitdepth;
int vlen;
double p;
double sf;
double  delta;
std::vector<int> wfData_rand, wfData_L;
double RowSt, RowEd, ColSt, ColEd;
Mat srcRGB, srcLAB, srcLABc, wmLAB, wmRGB, wmLABc, wmRGBc;
vector<Mat> srcLABc_cs, stgLABc_cs, srcLAB_cs;



