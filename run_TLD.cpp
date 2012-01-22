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
#include <iostream>

#include "tld/tld.h"

using namespace tld;

TldStruct tldStruct;

int main(int argc, char* argv[]) {

	std::string videopath = "";
	int camindex = -1, startFrame = 0;
	int x0 = -1, x1 = -1, y0 = -1, y1 = -1;
	unsigned int nodisplay = 0;
	cv::VideoCapture captureSource;


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
			if (i + 2 <= argc && x0 == -1 && x1 == -1) {
				x0 = std::min(double(atoi(argv[i + 1])), double(atoi(argv[i + 2])));
				x1 = std::max(double(atoi(argv[i + 1])), double(atoi(argv[i + 2])));
			}
		} else if (current == "-y") {
			if (i + 2 <= argc && y0 == -1 && y1 == -1) {
				y0 = std::min(double(atoi(argv[i + 1])), double(atoi(argv[i + 2])));
				y1 = std::max(double(atoi(argv[i + 1])), double(atoi(argv[i + 2])));
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
		std::cout << "No image source specified.\n";
		exit(0);
	}

	if (x0 == -1 || x1 == -1 || y0 == -1 || y1 == -1) {
		std::cout << "No bounding box specified.\n";
		exit(0);
	}

	if(camindex >= 0)
		captureSource.open(camindex);
	else if(!videopath.empty())
		captureSource.open(videopath);
	else
		assert(false && "No image source.");

	if(!captureSource.isOpened()) {
		std::cout << "Could not open video source.\n";
		exit(0);
	}

	CvCaptureImageSource imgsource(&captureSource);

	for(int i = 0; i < startFrame; i++) {
		// Ignore all frames up to startFrame.
		if(!imgsource.nextImage()) {
			std::cout << "Not enough frames to handle.\n";
			exit(0);
		}
	}

	tldInitDefaultTldStruct(tldStruct);
	tldSetImageSource(tldStruct, &imgsource);
	tldSetBB(tldStruct, x0, y0, x1 - x0, y1 - y0);

	tldExample(tldStruct, !nodisplay);
}
