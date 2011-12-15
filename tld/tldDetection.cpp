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
#include "../mex/mex.h"
#include <vector>
#include <algorithm>
#include <limits>

/**
 * Used to get sorted indices
 */
struct Indices {
	double conf;
	int index;

	Indices() :
		conf(0), index(0) {
	}
	Indices(double con, int idx) :
		conf(con), index(idx) {
	}

	static bool compare(const Indices& a, const Indices& b) {
		return a.conf > b.conf;
	}
};

/**
 * Detects new patches with nearest Neighbor and returns
 * a number of confidential bounding boxes.
 *
 * @param tld learned structure
 * @param i index
 * @param dBB reference to confidential bounding boxes
 * @return confidential values
 *
 */
Eigen::Matrix<double, 20, 1> tldDetection(TldStruct& tld, int i, Eigen::Matrix<
		double, 4, 20>& dBB, int& n) {

	dBB = Eigen::Matrix<double, 4, 20>::Constant(4, 20, std::numeric_limits<
			double>::quiet_NaN());

	Eigen::Matrix<double, 20, 1> confi = Eigen::Matrix<double, 20, 1>::Zero(20);

	// evaluates Ensemble Classifier: saves sum of posteriors to 'tld.tmp.conf', saves
	// measured codes to 'tld.tmp.patt'
	fern4(tld.currentImg, tld.control.maxbbox, tld.var, tld.tmp.conf,
			tld.tmp.patt);

	// get indexes of bounding boxes that passed through the Ensemble Classifier
	std::vector<int> idx_dt;
	for (int j = 0; j < tld.tmp.conf.size(); j++) {
		if (tld.tmp.conf(j) > tld.model.num_trees * tld.model.thr_fern)
			idx_dt.push_back(j);
	}

	// Get max 100 bounding boxes with best confidence
	if (idx_dt.size() > 100) {
		std::vector<Indices> idx;
		for (unsigned int i = 0; i < idx_dt.size(); i++) {
			Indices id(tld.tmp.conf(idx_dt[i]), idx_dt[i]);
			idx.push_back(id);
		}
		std::sort(idx.begin(), idx.end(), Indices::compare);
		idx_dt.clear();
		for (int j = 0; j < 100; j++)
			idx_dt.push_back(idx[j].index);
	}
	Detection dt;

	// get the number detected bounding boxes so-far
	int num_dt = idx_dt.size();

	// if nothing detected, return
	if (num_dt == 0) {
		tld.dt = dt;
		n = 1;
		return confi;
	}

	// initialize detection structure
	std::vector<int> idxcopy;

	dt.nbb = num_dt;
	for (int j = 0; j < num_dt; j++) {
		dt.bb.col(j) = tld.grid.block(0, idx_dt[j], 4, 1);

		Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), 1> ex =
				tldGetPattern(tld.currentImg, dt.bb.col(j),
						tld.model.patchsize, 0); // measure patch

		Eigen::MatrixXd result = tldNN(ex, tld); // evaluate nearest neighbour classifier

		// fill detection structure
		if (result(0, 0) > tld.model.thr_nn)
			idxcopy.push_back(j);
		dt.conf2(j) = result(0, 1);
		dt.patch.col(j) = ex;
	}

	// get all indexes that made it through the nearest neighbour
	int idxsize = idxcopy.size();

	// if no conf bb was detected return nan
	if (idxsize == 0) {
		dBB = Eigen::Matrix<double, 4, 20>::Constant(4, 20,
				std::numeric_limits<double>::quiet_NaN());
		tld.dt = dt;
		n = 1;
		return confi;
	}
	// save detected bounding boxes
	//dBB.resize(4, idxsize);
	if (idxsize > 20)
		idxsize = 20;
	for (int o = 0; o < idxsize; o++)
		dBB.col(o) = dt.bb.col(idxcopy[o]);

	// conservative confidences
	//confi.resize(idxsize);
	for (int o = 0; o < idxsize; o++)
		confi(o) = dt.conf2(idxcopy[o]);

	// save the whole detection structure
	tld.dt = dt;
	n = idxsize;

	return confi;
}

