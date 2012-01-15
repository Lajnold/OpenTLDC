/**
 * OpenTLDC is an algorithm for tracking of unknown objects
 * in unconstrained video streams. It is based on TLD,
 * published by Zdenek Kalal
 * (see http://info.ee.surrey.ac.uk/Personal/Z.Kalal/tld.html).
 *
 * Copyright (C) 2011 Sascha Schrader, Stefan Brending, Adrian Block
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <iostream>
#include <limits>
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include "mex.h"

using namespace tld;

static const int WIN_SIZE = 4;

void euclideanDistance(CvPoint2D32f *point1, CvPoint2D32f *point2,
		float *match, int nPts) {

	for (int i = 0; i < nPts; i++) {

		match[i] = sqrt((point1[i].x - point2[i].x) * (point1[i].x
				- point2[i].x) + (point1[i].y - point2[i].y) * (point1[i].y
				- point2[i].y));

	}
}

void normCrossCorrelation(IplImage *prevImg, IplImage *currImg,
		CvPoint2D32f *points0, CvPoint2D32f *points1, int nPts, char *status,
		float *match, int winsize, int method) {

	IplImage *rec0 = cvCreateImage(cvSize(winsize, winsize), 8, 1);
	IplImage *rec1 = cvCreateImage(cvSize(winsize, winsize), 8, 1);
	IplImage *res = cvCreateImage(cvSize(1, 1), IPL_DEPTH_32F, 1);

	for (int i = 0; i < nPts; i++) {
		if (status[i] == 1) {
			cvGetRectSubPix(prevImg, rec0, points0[i]);
			cvGetRectSubPix(currImg, rec1, points1[i]);
			cvMatchTemplate(rec0, rec1, res, method);
			match[i] = ((float *) (res->imageData))[0];

		} else {
			match[i] = 0.0;
		}
	}
	cvReleaseImage(&rec0);
	cvReleaseImage(&rec1);
	cvReleaseImage(&res);

}

// Lucas-Kanade
Eigen::Matrix<double, 4, 150> lk2(TldStruct &tld, IplImage* prevImg, IplImage* currImg,
		Eigen::Matrix<double, 2, 150> const & pointsI, Eigen::Matrix<double, 2, 150> const & pointsJ,
		unsigned int sizeI, unsigned int sizeJ, unsigned int level) {

	double nan = std::numeric_limits<double>::quiet_NaN();

	int Level;
	if (level != 0) {
		Level = (int) level;
	} else {
		Level = 5;
	}

	int I = 0;
	int J = 1;
	int Winsize = 10;

	// Points
	int nPts = sizeI;

	if (nPts != sizeJ) {
		std::cout << "Inconsistent input!" << std::endl;
		return Eigen::MatrixXd::Zero(1, 1);
	}

	CvPoint2D32f* points[3];
	points[0] = (CvPoint2D32f*) cvAlloc(nPts * sizeof(CvPoint2D32f)); // template
	points[1] = (CvPoint2D32f*) cvAlloc(nPts * sizeof(CvPoint2D32f)); // target
	points[2] = (CvPoint2D32f*) cvAlloc(nPts * sizeof(CvPoint2D32f)); // forward-backward

	for (int i = 0; i < nPts; i++) {
		points[0][i].x = pointsI(0, i);
		points[0][i].y = pointsI(1, i);
		points[1][i].x = pointsJ(0, i);
		points[1][i].y = pointsJ(1, i);
		points[2][i].x = pointsI(0, i);
		points[2][i].y = pointsI(1, i);
	}

	IplImage *prevPyr = tld.lkData.pyramid;
	if(!prevPyr)
		prevPyr = cvCreateImage(cvGetSize(prevImg), 8, 1);

	IplImage *currPyr = cvCreateImage(cvGetSize(currImg), 8, 1);

	float *ncc = (float*) cvAlloc(nPts * sizeof(float));
	float *fb = (float*) cvAlloc(nPts * sizeof(float));
	char *status = (char*) cvAlloc(nPts);

	cvCalcOpticalFlowPyrLK(prevImg, currImg, prevPyr, currPyr, points[0],
			points[1], nPts, cvSize(WIN_SIZE, WIN_SIZE), Level, status, 0,
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
			CV_LKFLOW_INITIAL_GUESSES);
	cvCalcOpticalFlowPyrLK(currImg, prevImg, currPyr, prevPyr, points[1],
			points[2], nPts, cvSize(WIN_SIZE, WIN_SIZE), Level, 0, 0,
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
			CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY
					| CV_LKFLOW_PYR_B_READY );

	normCrossCorrelation(prevImg, currImg, points[0], points[1], nPts, status,
			ncc, Winsize, CV_TM_CCOEFF_NORMED);
	euclideanDistance(points[0], points[2], fb, nPts);

	// Output
	int M = 4;
	Eigen::MatrixXd output(M, 150);
	for (int i = 0; i < nPts; i++) {
		if (status[i] == 1) {
			output(0, i) = (double) points[1][i].x;
			output(1, i) = (double) points[1][i].y;
			output(2, i) = (double) fb[i];
			output(3, i) = (double) ncc[i];
		} else {
			output(0, i) = nan;
			output(1, i) = nan;
			output(2, i) = nan;
			output(3, i) = nan;
		}
	}

	cvFree(&points[0]);
	cvFree(&points[1]);
	cvFree(&points[2]);

	cvReleaseImage(&prevPyr);
	tld.lkData.pyramid = currPyr;

	return output;
}
