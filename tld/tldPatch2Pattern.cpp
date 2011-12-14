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

#include "tld.h"
/* Converts an image to Eigen Matrix */
Eigen::Matrix<double, PATCHSIZE * PATCHSIZE, 1> tldPatch2Pattern(CvImage patch,
		Patchsize const& patchsize) {

	CvImage dest = CvImage(
			cvSize((int) patchsize.x, (int) patchsize.y), patch.depth(),
			patch.channels());

	//bilinear' is faster
	cvResize(patch, dest);
	Eigen::MatrixXd pattern(patchsize.x * patchsize.y, 1);
	for (int x = 0; x < dest.width(); x++)
		for (int y = 0; y < dest.height(); y++)
			pattern(x*patchsize.x + y, 0) = double(((uchar*) (dest.data() + dest.step()
					* (y)))[x]);

	// calculate column-wise mean
	Eigen::RowVectorXd mean(patchsize.x);
	mean = pattern.colwise().mean();
	pattern = pattern.rowwise() - mean;
	return pattern;
}
