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

#include "mex.h"
#include <math.h>

#ifndef NAN
#define NAN 0/0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979L
#endif

// colwise access
#define nextrow(tmp, widthStep) ((tmp)+widthStep)
#define nextcol(tmp) ((tmp)+1)
#define nextr_c(tmp, widthStep) ((tmp)+widthStep+1)

/* Warps image of size w x h, using affine transformation matrix (2x2 part)
 and offset (center of warping) ofsx, ofsy. Result is the region of size
 defined with roi. */
void warp_image_roi(const cv::Mat &image, int w, int h, Eigen::Matrix3d const & H,
		double xmin, double xmax, double ymin, double ymax, double fill,
		double *result) {

	double curx, cury, curz, wx, wy, wz, ox, oy, oz;
	int x, y, widthStep = image.step;
	const unsigned char *tmp;
	double *output = result, i, j, xx, yy;
	/* precalulate necessary constant with respect to i,j offset
	 translation, H is column oriented (transposed) */
	ox = H(0, 2);
	oy = H(1, 2);
	oz = H(2, 2);

	yy = ymin;
	for (j = 0; j < (int) (ymax - ymin + 1); j++) {
		/* calculate x, y for current row */
		curx = H(0, 1) * yy + ox;
		cury = H(1, 1) * yy + oy;
		curz = H(2, 1) * yy + oz;
		xx = xmin;
		yy = yy + 1;
		for (i = 0; i < (int) (xmax - xmin + 1); i++) {
			/* calculate x, y in current column */
			wx = H(0, 0) * xx + curx;
			wy = H(1, 0) * xx + cury;
			wz = H(2, 0) * xx + curz;
			//       printf("%g %g, %g %g %g\n", xx, yy, wx, wy, wz);
			wx /= wz;
			wy /= wz;
			xx = xx + 1;

			x = (int) floor(wx);
			y = (int) floor(wy);

			if (x >= 0 && y >= 0) {
				wx -= x;
				wy -= y;
				if (x + 1 == w && wx == 1)
					x--;
				if (y + 1 == h && wy == 1)
					y--;
				if ((x + 1) < w && (y + 1) < h) {
					tmp = &image.ptr(y)[x];
					*output++ = (*(tmp) * (1 - wx) + *nextcol(tmp) * wx)
									* (1 - wy) + (*nextrow(tmp,widthStep) * (1 - wx)
									+ *nextr_c(tmp,widthStep) * wx) * wy;

				} else
					*output++ = fill;
			} else
				*output++ = fill;
		}
	}
}

cv::Mat toMat(const double *image, int num_cols, int num_rows) {

	cv::Mat result(num_rows, num_cols, CV_8UC1);
	const double* s_ptr = image;

	for (int i = 0; i < num_rows; i++) {
		unsigned char *row = result.ptr(i);
		for (int j = 0; j < num_cols; j++, s_ptr++) {
			row[j] = (unsigned char) (*s_ptr);
		}
	}

	return result;
}

cv::Mat warp(const cv::Mat & img, Eigen::Matrix3d const & H, Eigen::Vector4d const & box) {
	int w, h;
	double *result;
	double xmin, xmax, ymin, ymax, fill;
	w = img.cols;
	h = img.rows;

	xmin = box(0);
	xmax = box(1);
	ymin = box(2);
	ymax = box(3);

	fill = 0;

	result = new double[((int) (xmax - xmin + 1) * (int) (ymax - ymin + 1))];
	warp_image_roi(img, w, h, H, xmin, xmax, ymin, ymax, fill, result);

	cv::Mat out = toMat(result, (int) (xmax - xmin + 1), (int) (ymax - ymin + 1));

	delete[] result;

	return out;

}
