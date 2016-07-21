#include "dctTest.h";
#include <vector>;
#include<opencv2/opencv.hpp>;
#include<math.h>
#include<opencv2/opencv.hpp>;
#include<math.h>
#include<fstream>
#include<string>
#include<iostream>
#include"stdlib.h"
using namespace cv;
using namespace std;
double quantificate(double z, int d, double delta) {
	double dither = d*delta / 2;
	double z1 = delta * round((z + dither) / delta) - dither;
	return z1;
}

int minEucDistance(double z1, double delta) {
	double m1;
	double q00 = quantificate(z1, 0, delta);
	double q10 = quantificate(z1, 1, delta);
	double zz[2] = { std::abs(z1 - q00),std::abs(z1 - q10) };
	if (zz[0]<zz[1]) {
		m1 = 0;
	}
	if (zz[0]>zz[1]) {
		m1 = 1;
	}
	return m1;
}

Mat revDiagCoeff(Mat newMtr, vector<double> V, int vlen) {
	int	m1 = newMtr.rows;
	int n1 = newMtr.cols;
	if (m1 != n1) {
		cout << "请输入方阵" << endl;
	}
	int rowStart = vlen - 1;
	int p = 0;
	while (rowStart<m1)
	{
		int i = rowStart;
		int j = 0;
		while (i >= 0)
		{
			newMtr.at<double>(i, j) = V[p];
			i = i - 1;
			j = j + 1;
			p = p + 1;
		}
		rowStart = rowStart + 1;
	}
	int colStart = 1;
	while (colStart<(n1 - vlen + 1))
	{
		int i = m1 - 1;
		int j = colStart;
		while (j<n1)
		{
			newMtr.at<double>(i, j) = V[p];
			i = i - 1;
			j = j + 1;
			p = p + 1;
		}
		colStart = colStart + 1;
	}
		
	return newMtr;
}
vector<double> getPowMat(vector<double> c, int p, int start, int end) {
	vector<double> c1;
	for (int i=start-1;i<=end-1;i++) 
	{
		c1.push_back(std::pow(c[i], p));
	}
	return c1;
}
double getLxOrLy(vector<double> cc, int vlen, double p1) {
	double sum = 0;
	for (int i = 0; i < cc.size(); i++) {
		sum = sum + cc[i];
	}
	double sum1 = std::abs(sum);
	double sum2 = sum1 / vlen;
	return std::pow(sum2, p1);
}
Mat gauLowPass(Mat cfData, int n, double sigma) {
	Mat glData;
	GaussianBlur(cfData, glData, cv::Size(n, n), sigma);
	return glData;
}
vector<double> getDiagCoeff(Mat Mtr, int vlen) {
	int m1 = Mtr.rows;
	int n1 = Mtr.cols;
	vector<double> V;
	if (m1 != n1) {
		cout << "请输入方阵" << endl;
	}
	int rowStart = vlen-1;
	while (rowStart<m1)
	{
		int i = rowStart;
		int j = 0;
		while (i>=0) 
		{
			V.push_back(Mtr.at<double>(i,j));
			i = i - 1;
			j = j + 1;
		}
		rowStart = rowStart + 1;
	}
	int colStart = 1;
	while (colStart<(n1 - vlen + 1))
	{
		int i = m1-1;
		int j = colStart;
		while (j<n1)
		{
			V.push_back(Mtr.at<double>(i,j));
			i = i - 1;
			j = j + 1;
		}
		colStart = colStart + 1;
	}
	return V;
}
vector<int> readMsg(String dir, int msglen) {
	ifstream in(dir);
	vector<int> wfData;
	string s;
	int i = 1;
	while (getline(in, s) && (i <= msglen)) {
		int ss = atoi(s.c_str());
		wfData.push_back(ss);
		i++;
	}
	return wfData;
}
void initWaterMark() {
	msgLen = 128;
	bitdepth = 8;
	vlen = 32;
	sf = 0.99;
	p = 2.0;
	delta = 0.33;
	srcRGB = imread("1.tif");
	cvtColor(srcRGB, srcLAB, CV_BGR2Lab);
	srcLAB = srcLAB*sf;
	m = srcLAB.cols;//获取图像矩阵的列数
	n = srcLAB.rows;//行数
	t = 3;//通道数
	RowSt = 66;
	RowEd = 515;
	ColSt = 66;
	ColEd = 515;
	//选取嵌入区域
	srcLAB(Range(RowSt - 1, RowEd), Range(ColSt - 1, ColEd)).copyTo(srcLABc);
	split(srcLABc, srcLABc_cs);
	//读入水印信息
	wfData_rand = readMsg("msg.txt", msgLen);
	wfData_L = readMsg("msg_LAB_delta0.33.txt", msgLen);
}


Mat giQimHide_DCT_Glp2(Mat cfData, vector<int> wfData, double delta,
	int vlen, double p, char type) {
	Mat cfL = gauLowPass(cfData, 3, 0.5);//低频部分	
	Mat cfH = cfData - cfL;//高频部分
	Mat target, targetRest;
	if (type== 'L' )
	{
		target = cfL;
		targetRest = cfH;
	}
	if (type == 'H')
	{
		target = cfH;
		targetRest = cfL;
	}
	int si[2] = { target.rows,target.cols };//获取矩阵target的行数和列数
	int len = wfData.size();
	int N = 2 * (si[0] - vlen) + 1;
	if (len < N)
	{
		for (int i = len; i < N; i++) {
			wfData.push_back(0);
		}
	}
	if (len > N)
	{
		wfData.erase(wfData.begin() + N, wfData.end());
	}
	Mat tempDct, Dct;
	target.convertTo(tempDct,CV_64FC1);
	dct(tempDct, Dct);
	vector<double> V = getDiagCoeff(Dct, vlen);
	int k = vlen;
	int startPosX = 1;
	int	endPosX, startPosY, endPosY, vecLen;
	for (int i = 1; i <= N;i++) {
		int d = wfData[i-1];
		if (k<=si[0])
		{
			vecLen = floor(k / 2);
		}
		else {
			vecLen = floor((2 * si[0] - k) / 2);
		}
		endPosX = startPosX + vecLen - 1;
		if ((k%2)==0)
		{
			startPosY = endPosX + 1;
		}
		else {			
			startPosY = endPosX + 2;
		}
		endPosY = startPosY + vecLen - 1;
		vector<double> v1Col = getPowMat(V, p, startPosX, endPosX);
		vector<double> v2Col = getPowMat(V, p, startPosY, endPosY);
		double lx = getLxOrLy(v1Col, vecLen, 1 / p);
		double ly = getLxOrLy(v2Col, vecLen, 1 / p);
		double z = lx / ly;
		if ((std::abs(lx) <= 1e-6) || (std::abs(ly) <= 1e-6))
			continue;
		double z1 = quantificate(z, d, delta);
		if (z1 == 0)
		{
			z1 = delta / 8;
		}
		if (z1 < 0)
		{
			z1 = z1 + delta;
		}
		double r=std::sqrt(z1/z);
		for (int j = startPosX-1;j < endPosX; j++)
		{
			V[j] = r*V[j];
		}
		for (int j = startPosY-1; j < endPosY; j++)
		{
			V[j] = V[j] / r;
		}
		k = k + 1;
		startPosX = endPosY+1;
	}
	Mat revDct = revDiagCoeff(Dct, V, vlen);
	Mat O,O1;
	idct(revDct, O);
	O.convertTo(O1, CV_8UC1);
	Mat stg =O1+targetRest;
	return stg;
}
void embedWaterMrak() {
	for (int i = 0; i < 3; i++) {
		if (i == 0) {
			Mat temp = giQimHide_DCT_Glp2(srcLABc_cs[i], wfData_L, delta, vlen, p, 'H');
			stgLABc_cs.push_back(giQimHide_DCT_Glp2(temp, wfData_rand, delta, vlen, p, 'L'));
		}
		else
		{
			stgLABc_cs.push_back(srcLABc_cs[i]);
		}
	}
	merge(stgLABc_cs, wmLABc);
	wmLABc.copyTo(srcLAB(Range(RowSt - 1, RowEd), Range(ColSt - 1, ColEd)));
	cvtColor(srcLAB, wmRGB, CV_Lab2BGR);
}
std::vector<int> giQimDeHide_DCT_Glp2(Mat cfData, double delta,
	int vlen, double p, int wfLen, char type) {
	std::vector<int> o;
	Mat cfL = gauLowPass(cfData, 3, 0.5);
	Mat cfH = cfData - cfL;
	Mat target, targetRest;
	if (type == 'L')
	{
		target = cfL;
		targetRest = cfH;
	}
	if (type == 'H')
	{
		target = cfH;
		targetRest = cfL;
	}
	int si[2] = { target.rows,target.cols };
	//cout << "target.rows=" << target.rows << endl;
	//cout << "target.cols=" << target.cols << endl;
	int N = 2 * (si[0] - vlen) + 1;
	//cout << "N=" << N << endl;
	Mat tempDct, Dct;
	//cout << target.row(0) << endl;
	target.convertTo(tempDct, CV_64FC1);     //可能有问题，tempDct和Dct都imshow（）显示不了，不知道为什么
											 //cout << tempDct.row(0) << endl;
	dct(tempDct, Dct);
	//cout << Dct.row(0) << endl;

	vector<double> V = getDiagCoeff(Dct, vlen);
	//cout << V << endl;
	int k = vlen;
	int startPosX = 1;
	int startPosY;
	int	endPosX, endPosY, vecLen;

	for (int i = 1; i <= N; i++) {
		if (k <= si[0])
		{
			vecLen = floor(k / 2);
		}
		else {
			vecLen = floor((2 * si[0] - k) / 2);
		}
		endPosX = startPosX + vecLen - 1;
		if ((k % 2) == 0)
		{
			startPosY = endPosX + 1;
		}
		else {
			startPosY = endPosX + 2;
		}
		endPosY = startPosY + vecLen - 1;
		vector<double> v1Col = getPowMat(V, p, startPosX, endPosX);
		vector<double> v2Col = getPowMat(V, p, startPosY, endPosY);
		double lx = getLxOrLy(v1Col, vecLen, 1 / p);
		double ly = getLxOrLy(v2Col, vecLen, 1 / p);
		double z = lx / ly;
		if ((std::abs(lx) <= 1e-6) || (std::abs(ly) <= 1e-6))
			continue;
		o.push_back(minEucDistance(z, delta));
		k = k + 1;
		startPosX = endPosY+1;
		if (k>(2 * si[0] - 1))
		{
			cout << "第%d条对角线不存在,已超出范围!" << endl;
		}
	}
	if (wfLen < N) {
		o.erase(o.begin() + wfLen, o.end());
	}

	return o;
}
void main() {
	int sum = 0;
	initWaterMark();
	embedWaterMrak();
	vector<int> wm = giQimDeHide_DCT_Glp2(stgLABc_cs[0], 0.33, 32, 2.0, 128, 'L');
	for (int l = 0; l < wm.size();l++) {
		
		if (wm[l]== wfData_rand[l]) {
			sum++;
		}
	}
	cout << "Esum=:" << sum << endl;
	//imshow("srcRGB", srcRGB);
	//imshow("wmRGB", wmRGB);
	//imwrite("wmRGB.tif", wmRGB);
	//waitKey(0);
	
	system("pause");
}