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
#include "../img/img.h"
#include "../utils/utility.h"

/**
 * Pickups bbox and converts to Eigen matrix.
 *
 * @param img current img
 * @param bb matrix of bounding boxes
 * @param patchsize size of one patch
 * @param flip use mirrored patches yes / no
 *
 * @return patterns
 */
Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Eigen::Dynamic> tldGetPattern(
		ImgType& img, Eigen::Matrix<double, 4, Eigen::Dynamic> const & bb,
		Patchsize& patchsize, unsigned int flip) {

	// get patch under bounding box (bb), normalize it size,
	// reshape to a column
	// vector and normalize to zero mean and unit variance (ZMUV)

	// initialize output variable
	unsigned int nBB = bb.cols();
	Eigen::MatrixXd pattern = Eigen::MatrixXd::Zero(patchsize.x * patchsize.y,
			nBB);

	// for every bounding box
	for (unsigned int i = 0; i < nBB; i++) {

		// sample patch
		Eigen::Vector4d tmpbb = bb.col(i);
		CvImage patch = img_patch(img.input, tmpbb);

		// flip if needed
		if (flip) {
			cvFlip(patch, NULL, 1);
		}

		// normalize size to 'patchsize' and nomalize intensities to ZMUV
		pattern.col(i) = tldPatch2Pattern(patch, patchsize);
	}

	return pattern;

}
