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

#include <stdio.h>
#include <math.h>
#include <vector>
using namespace std;
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include "mex.h"

double bb_overlaphelper(Eigen::Vector4d const & bb1, Eigen::Vector4d const & bb2) {
	if (bb1(0) > bb2(2)) {
		return 0.0;
	}
	if (bb1(1) > bb2(3)) {
		return 0.0;
	}
	if (bb1(2) < bb2(0)) {
		return 0.0;
	}
	if (bb1(3) < bb2(1)) {
		return 0.0;
	}

	double colInt = min(bb1(2), bb2(2)) - max(bb1(0), bb2(0)) + 1;
	double rowInt = min(bb1(3), bb2(3)) - max(bb1(1), bb2(1)) + 1;

	double intersection = colInt * rowInt;
	double area1 = (bb1(2) - bb1(0) + 1) * (bb1(3) - bb1(1) + 1);
	double area2 = (bb2(2) - bb2(0) + 1) * (bb2(3) - bb2(1) + 1);

	return intersection / (area1 + area2 - intersection);
}

Eigen::VectorXd bb_overlap(Eigen::Matrix<double, 4, Eigen::Dynamic> const & bb1) {

	double nBB = bb1.cols();

	Eigen::VectorXd out(nBB * (nBB - 1) / 2);

	int counter = 0;
	for (int i = 0; i < nBB - 1; i++) {
		for (int j = i + 1; j < nBB; j++) {
			out(counter) = bb_overlaphelper(bb1.col(i), bb1.col(j));
			counter++;
		}
	}
	return out;
}

Eigen::MatrixXd bb_overlap(
		Eigen::Matrix<double, 4, Eigen::Dynamic> const & bb, int n1,
		Eigen::Matrix<double, 4, Eigen::Dynamic> const & bb1) {

	int N = bb.cols();
	int NN = n1;

	if (N == 0 || NN == 0) {
		N = 0;
		NN = 0;
	}

	Eigen::MatrixXd out(N, NN);


	for (int i = 0; i < N; i++) {
		for (int j = 0; j < NN; j++) {
			out(i, j) = bb_overlaphelper(bb.col(i), bb1.col(j));
		}
	}

	return out;
}

Eigen::MatrixXd bb_overlap(
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> const & bb,
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> const & bb1, int x) {

	int N = bb.cols();
	int NN = bb1.cols();
	Eigen::VectorXd out(N);

	if (x == 1) {
		for (int j = 0; j < N; j++) {
			out(j) = bb_overlaphelper(bb.col(j), bb1.col(j));
		}
	} else {
		for (int j = 0; j < N; j++) {
			double maxOvrlp = 0;
			for (int i = 0; i < NN; i++) {
				double overlap = bb_overlaphelper(bb.col(j), bb1.col(i));
				if (overlap > maxOvrlp) {
					maxOvrlp = overlap;
					out(j) = i + 1.0;
				}
			}
		}
	}
	return out;
}

