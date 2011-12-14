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

#ifndef IMG_H_
#define IMG_H_

#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "../tld/tld.h"

IplImage* img_patch(IplImage* img, Eigen::Vector4d const & bb, double randomize,
		p_par& p_par);

IplImage* img_patch(IplImage* img, Eigen::Vector4d const & bb);

IplImage* img_blur(IplImage* image, int sigma);
IplImage* img_blur(IplImage* image);
/*Initialize*/
void img_init(Config& cfg);
/*Get next image (grayscale)*/
IplImage* img_get();
/*Get next image (color)*/
IplImage* img_get_colored();

#endif /* IMG_H_ */
