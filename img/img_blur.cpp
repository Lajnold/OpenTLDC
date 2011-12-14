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
#include "img.h"

//apply a gaussian filter to the image
CvImage img_blur(CvImage image, int sigma){
	int csize = 6* sigma;
	CvImage out = CvImage(image.size(), 8, 1);
	cvSmooth(image, out, CV_GAUSSIAN, csize, csize);
	return out;
}

//apply a gaussian filter to the image
CvImage img_blur(CvImage image){
	int csize = 7;
	CvImage out = CvImage(image.size(), 8, 1);
	cvSmooth(image, out, CV_GAUSSIAN, csize, csize);
	return out;
}


