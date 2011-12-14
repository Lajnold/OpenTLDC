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

#include <iostream>
#include "img.h"

IplImage* img = 0;
IplImage* grayimg = 0;
IplImage* colored = 0;
CvCapture* cap = 0;

//get current image
IplImage* img_get() {
	img = cvQueryFrame(cap);
	cvReleaseImage(&colored);
	colored = cvCloneImage(img);
	grayimg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

	cvCvtColor(img, grayimg, CV_BGR2GRAY);

	return grayimg;
}

IplImage* img_get_colored() {
	return colored;
}
//Initialize image retrieval
void img_init(Config& cfg) {
	//img from cam
	if (cfg.camindex != -1)
		cap = cvCaptureFromCAM(cfg.camindex);
	//img from a path
	else
		cap = cvCaptureFromAVI(cfg.videopath.c_str());

	//cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, 1048);

	//ignore the first cnt frames
	unsigned int cnt = 0;
	while (cnt < cfg.startFrame) {
		img = cvQueryFrame(cap);
		cnt++;
	}
}
