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
#include <iostream>
#include "../bbox/bbox.h"
#include "../img/img.h"
#include "../mex/mex.h"
#include <limits>

void tldProcessFrame(TldStruct& tld, unsigned long i) {

	tld.prevImg = tld.currentImg;

	tld.cfg->imgsource->nextImage();

	//Input image
	ImgType im0;
	im0.input = tld.cfg->imgsource->getGrayImage();

	//Blurred image
	im0.blur = img_blur(im0.input);

	//color image with 3 channels
	tld.currentImg = im0;

	//switch from current to previous
	/* bbox */
	tld.prevBB = tld.currentBB;
	tld.currentBB = Eigen::Vector4d::Constant(
			std::numeric_limits<double>::quiet_NaN());

	/* valid */
	tld.prevValid = tld.currentValid;
	tld.currentValid = 0;

	//TRACKER
	//rame-to-frame tracking (MedianFlow)
	Eigen::VectorXd tldTrack = tldTracking(tld, tld.prevBB, i - 1, i);

	//DETECTOR
	//detect appearances by cascaded detector (variance filter -> ensemble classifier -> nearest neighbor)
	Eigen::Matrix<double, 4, 20> dBB;
	int nD = 0;
	Eigen::Matrix<double, 20, 1> detConf = tldDetection(tld, i, dBB, nD);

	//INTEGRATOR
	//Tracker defined?
	unsigned int TR = 1;
	if (isnan(tldTrack(0))) {
		TR = 0;
	}
	//Detector defined?
	unsigned int DT = 1;
	if (isnan(dBB(0, 0))) {
		DT = 0;
	}
	if (TR) {
		//copy tracker's result
		tld.size = 1;
		if (DT) {
			//cluster detections
			Eigen::MatrixXd cluster = bb_cluster_confidence(dBB, detConf, nD);

			//get indexes of all clusters that are far from tracker and are more confident than the tracker
			unsigned int len = cluster.cols() / 3;
			Eigen::MatrixXd cBB(4, len);
			cBB = cluster.block(0, 0, 4, len);
			Eigen::MatrixXd overlap = bb_overlap(tld.currentBB, len, cBB);

			Eigen::MatrixXd cConf(1, len);
			cConf = cluster.block(0, len, 1, len);
			Eigen::MatrixXd cSize(1, len);
			cSize = cluster.block(0, 2 * len, 1, len);
			std::vector<unsigned int> id;
			for (unsigned int j = 0; j < len; j++)
				if (overlap(0, j) < 0.5 && cConf(0, j) > tld.conf)
					id.push_back(j);
			//if there is ONE such cluster, re-initialize the tracker
			if (id.size() == 1) {
				tld.currentBB = cBB.col(id[0]);
				tld.conf = cConf(0, id[0]);
				tld.size = cSize(0, id[0]);
				tld.currentValid = 0;
			} else {
				//adjust the tracker's trajectory
				//get indexes of close detections
				overlap = bb_overlap(tldTrack.topRows(4), tld.dt.nbb, tld.dt.bb);
				std::vector<unsigned int> idTr;
				for (int p = 0; p < overlap.cols(); p++)
					if (overlap(0, p) > 0.7)
						idTr.push_back(p);
				//weighted average trackers trajectory with the close detections
				Eigen::MatrixXd meanmat(4, 10 + idTr.size());
				for (int p = 0; p < 10; p++) {
					meanmat.col(p) = tldTrack.topRows(4);
				}
				for (unsigned int p = 10; p < 10 + idTr.size(); p++)
					meanmat.col(p) = tld.dt.bb.col(idTr[p - 10]);
				tld.currentBB = meanmat.rowwise().mean();
			}
		}
	} else {
		if (DT) {
			//cluster detections
			Eigen::MatrixXd cluster = bb_cluster_confidence(dBB, detConf, nD);
			//if there is just a single cluster, re-initialize the tracker
			if (cluster.cols() / 3 == 1) {
				tld.currentBB = cluster.col(0);
				tld.conf = cluster(0, 1);
				tld.size = cluster(0, 2);
				tld.currentValid = 0;
			}
		}
	}

	//LEARNING
	if (tld.control.update_detector && tld.currentValid == 1)
		tldLearning(tld, i);
	std::cout << "BB - xmin: " << tld.currentBB(0) << " ymin: "
			<< tld.currentBB(1) << " xmax: " << tld.currentBB(2) << " ymax: "
			<< tld.currentBB(3) << std::endl;
}
