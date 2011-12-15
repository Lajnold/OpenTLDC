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

#include <cassert>

#include "tld/tld.h"

TldStruct tld;

int main(int argc, char* argv[]) {

	std::string videopath = "";
	int camindex = -1, startFrame = 0;
	Eigen::Vector4d initBB;
	initBB << -1, -1, -1, -1;
	unsigned int nodisplay = 0;
	CvCapture *captureSource;


	for (int i = 1; i < argc; ++i) {

		std::string current = argv[i];
		if (current == "-vid") {
			if (i + 1 <= argc && camindex == -1) {
				videopath = argv[i + 1];
				i++;
				continue;
			}
		} else if (current == "-cam") {
			if (i + 1 <= argc && videopath == "") {
				camindex = atoi(argv[i + 1]);
				i++;
				continue;
			}
		} else if (current == "-x") {
			if (i + 2 <= argc && initBB(0) == -1 && initBB(2) == -1) {
				initBB(0) = std::min(double(atoi(argv[i + 1])), double(atoi(
						argv[i + 2])));

				initBB(2) = std::max(double(atoi(argv[i + 1])), double(atoi(
						argv[i + 2])));
			}
		} else if (current == "-y") {
			if (i + 2 <= argc && initBB(1) == -1 && initBB(3) == -1) {
				initBB(1) = std::min(double(atoi(argv[i + 1])), double(atoi(
						argv[i + 2])));

				initBB(3) = std::max(double(atoi(argv[i + 1])), double(atoi(
						argv[i + 2])));
			}
		} else if (current == "-nodisplay") {
			nodisplay = 1;
		} else if (current == "-fr") {
			if (i + 1 <= argc) {
				startFrame = atoi(argv[i + 1]);
				i++;
				continue;
			}
		}
	}

	if (videopath == "" && camindex == -1) {
		exit(0);
	}

	if (initBB(0) == -1 || initBB(1) == -1 || initBB(2) == -1 || initBB(3)
			== -1)
		exit(0);

	if(camindex >= 0)
		captureSource = cvCaptureFromCAM(camindex);
	else if(!videopath.empty())
		captureSource = cvCaptureFromAVI(videopath.c_str());
	else
		assert(false && "No image source.");

	CvCaptureImageSource imgsource(captureSource);

	for(int i = 0; i < startFrame; i++) {
		// Ignore all frames up to startFrame.
		if(!imgsource.nextImage()) {
			cvReleaseCapture(&captureSource);
			exit(0); // Ran out of images.
		}
	}

	tldInitDefaultTldStruct(tld);
	tldSetImageSource(tld, &imgsource);
	tldSetBB(tld, initBB);

	for(int i = 0; i < startFrame; i++) {
		// Ignore all frames up to startFrame.
		if(!tld.cfg.imgsource->nextImage())
			exit(0); // Ran out of images.
	}

	tldExample(tld, !nodisplay);
	cvReleaseCapture(&captureSource);
}
