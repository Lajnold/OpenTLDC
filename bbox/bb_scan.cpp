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

#include "bbox.h"
#include "../utils/utility.h"

void bb_scan(tld::TldStruct& tld, Eigen::Vector4d const & bb,
		Eigen::Vector2i imsize, int minwin) {
	double shift = 0.1;
	//used for scaling the bb
	Eigen::VectorXd scale(21);
	scale << -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
	for (unsigned int i = 0; i < 21; i++)
		scale(i) = pow(1.2, scale(i));
	int minBB = minwin;

	if (bb_width(bb) < minwin)
		return;

	Eigen::VectorXd bbW(21);
	//scale bb on x axis
	for (unsigned int x = 0; x < 21; x++) {
		bbW(x) = floor((bb_width(bb) * scale(x)) + 0.5);
	}

	Eigen::VectorXd bbH(21);
	//scale bb on y axis
	for (unsigned int x = 0; x < 21; x++) {
		bbH(x) = floor((bb_height(bb) * scale(x)) + 0.5);
	}

	Eigen::VectorXd bbSHH(21);
	Eigen::VectorXd bbSHW(21);
	//shift the scales
	for (unsigned int x = 0; x < 21; x++)
		bbSHH(x) = bbH(x) * shift;

	for (unsigned int x = 0; x < 21; x++) {
		if (bbH(x) <= bbW(x))
			bbSHW(x) = bbH(x) * shift;
		else
			bbSHW(x) = bbW(x) * shift;
	}

	Eigen::VectorXd bbF(4);
	bbF << 2, 2, imsize(0), imsize(1);
	Eigen::Matrix<double, 6, Eigen::Dynamic> bbs;
	Eigen::Matrix<double, 6, Eigen::Dynamic> bbsbak;
	Eigen::MatrixXd sca(2, 21);
	//create a grid of bounding boxes with different scales
	for (unsigned int i = 0; i < 21; i++) {
		if (bbW(i) < minBB || bbH(i) < minBB)
			continue;
		Eigen::VectorXd left(1);
		Eigen::VectorXd leftbak(1);
		Eigen::VectorXd top(1);
		Eigen::VectorXd topbak(1);
		double val;

		val = bbF(0);
		for (unsigned int p = 0; val <= bbF(2) - bbW(i) - 1; val += bbSHW(i), p++) {
			leftbak.resize(left.size());
			leftbak = left;
			left.resize(p + 1);
			if (p > 0)
				left << leftbak, floor(val + 0.5);
			else
				left(0) = floor(val + 0.5);
		}

		val = bbF(1);
		for (unsigned int p = 0; val <= bbF(3) - bbH(i) - 1; val += bbSHH(i), p++) {
			topbak.resize(top.size());
			topbak = top;
			top.resize(p + 1);
			if (p > 0)
				top << topbak, floor(val + 0.5);
			else
				top(0) = floor(val + 0.5);
		}

		Eigen::MatrixXd grid(2, top.size() * left.size());

		unsigned int cnt = 0;
		for (int k = 0; k < left.size(); k++)
			for (int w = 0; w < top.size(); w++) {
				grid(0, cnt) = top(w);
				grid(1, cnt) = left(k);
				cnt++;
			}

		Eigen::MatrixXd bbsnew(6, grid.cols());
		bbsnew.row(0) = grid.row(1);
		bbsnew.row(1) = grid.row(0);
		bbsnew.row(2) = grid.array().row(1) + bbW(i) - 1;
		bbsnew.row(3) = grid.array().row(0) + bbH(i) - 1;
		bbsnew.row(4) = Eigen::MatrixXd::Constant(1, grid.cols(), i + 1);
		bbsnew.row(5) = Eigen::MatrixXd::Constant(1, grid.cols(), left.size());
		bbsbak.resize(6, bbs.cols());
		bbsbak = bbs;
		bbs.resize(6, bbs.cols() + bbsnew.cols());
		//lets have some fun!
		if (i > 0)
			bbs << bbsbak, bbsnew;
		else
			bbs = bbsnew;
		//save the scales on x and y axis
		sca(0, i) = bbH(i);
		sca(1, i) = bbW(i);
	}
	tld.grid = bbs;
	tld.nGrid = bbs.cols();
	tld.scales = sca;
}
