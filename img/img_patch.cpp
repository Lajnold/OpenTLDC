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

#include <cmath>

#include "img.h"
#include "../utils/utility.h"
#include "../bbox/bbox.h"
#include "../mex/mex.h"

CvImage img_patch(CvImage img, Eigen::Vector4d const & bb, double randomize,
		tld::P_par& p_par) {
	if (randomize <= 0)
		return CvImage();

	unsigned int NOISE = p_par.noise;
	unsigned int ANGLE = p_par.angle;
	double SCALE = p_par.scale;
	double SHIFT = p_par.shift;

	Eigen::Vector2d cp;
	cp = bb_center(bb);
	Eigen::Matrix3d Sh1;
	Sh1 << 1, 0, -(cp(0) - 1), 0, 1, -(cp(1) - 1), 0, 0, 1;

	double sca = 1 - SCALE * (uniform() - 0.5);
	Eigen::Matrix3d Sca;
	Sca << sca, 0, 0, 0, sca, 0, 0, 0, 1;

	double ang = 2 * M_PI / 360 * ANGLE * (uniform() - 0.5);
	double ca = cos(ang);
	double sa = sin(ang);

	Eigen::Matrix3d Ang;
	Ang << ca, -sa, 0, sa, ca, 0, 0, 0, 1;

	double shR = SHIFT * bb_height(bb) * (uniform() - 0.5);
	double shC = SHIFT * bb_width(bb) * (uniform() - 0.5);
	Eigen::Matrix3d Sh2;
	Sh2 << 1, 0, shC, 0, 1, shR, 0, 0, 1;

	double bbW = bb_width(bb) - 1;
	double bbH = bb_height(bb) - 1;
	Eigen::Vector4d box;
	box << -bbW / 2, bbW / 2, -bbH / 2, bbH / 2;
	//
	//	    H     = Sh2*Ang*Sca*Sh1;
	Eigen::Matrix3d H;
	H = Sh2 * Ang * Sca * Sh1;
	//	    bbsize = bb_size(bb);
	BBOXSIZE bbsize = bb_size(bb);
	CvImage patch = warp(img, H.inverse(), box);
	//	    patch = uint8(warp(img,inv(H),box) + NOISE*randn(bbsize(1),bbsize(2)));
	Eigen::MatrixXd noisy = Eigen::MatrixXd::Random(bbsize.height, bbsize.width);
	noisy = noisy.normalized() * NOISE;

	for (int y = 0; y < patch.height(); y++)
		for (int x = 0; x < patch.width(); x++)
			((uchar*) (patch.data() + patch.step() * (y)))[x]
					+= noisy(y, x);

	return patch;
}

CvImage img_patch(CvImage img, Eigen::Vector4d const & bb) {

	Eigen::Vector4d rounded;
	rounded(0) = floor(bb(0) + 0.5);
	rounded(1) = floor(bb(1) + 0.5);
	rounded(2) = floor(bb(2) + 0.5);
	rounded(3) = floor(bb(3) + 0.5);

	//	 % All coordinates are integers
	//	    if sum(round(bb)-bb)==0
	if ((rounded - bb).sum() == 0) {
		unsigned int L = std::min(std::max(0, int(bb(0))), img.width());
		unsigned int T = std::min(std::max(0, int(bb(1))), img.height());
		unsigned int R = std::min(img.width() - 1, int(bb(2)));
		unsigned int B = std::min(img.height() - 1, int(bb(3)));
		CvImage patch = CvImage(cvSize((int) (R - L + 1), (int) (B - T + 1)), IPL_DEPTH_8U, 1);

		//	        patch = img(T:B,L:R);
		for (unsigned int y = T; y <= B; y++)
			for (unsigned int x = L; x <= R; x++)
				((uchar*) (patch.data() + patch.step() * (y - T)))[x
						- L]
						= ((uchar*) (img.data() + img.step() * (y)))[x];

		return patch;
	} else {


		//
		//	        % Sub-pixel accuracy
		//
		//	        cp = 0.5 * [bb(1)+bb(3); bb(2)+bb(4)]-1;
		Eigen::Vector2d cp;
		cp = bb_center(bb);
		//	        H = [1 0 -cp(1); 0 1 -cp(2); 0 0 1];
		Eigen::Matrix3d H;
		H << 1, 0, -(cp(0) - 1), 0, 1, -(cp(1) - 1), 0, 0, 1;
		//
		//	        bbW = bb(3,:)-bb(1,:);
		double bbW = bb(2) - bb(0);
		//	        bbH = bb(4,:)-bb(2,:);
		double bbH = bb(3) - bb(1);
		//	        if bbW <= 0 || bbH <= 0
		if (bbW < 0 || bbH < 0)
			return CvImage();
		//	            patch = [];
		//	            return;
		//	        end
		//	        box = [-bbW/2 bbW/2 -bbH/2 bbH/2];
		Eigen::Vector4d box;
		box << -bbW / 2, bbW / 2, -bbH / 2, bbH / 2;
		//

		// We always use one channel in color space
		//	            patch = warp(img,inv(H),box);

		CvImage patch = warp(img, H.inverse(), box);
		return patch;

	}

}
