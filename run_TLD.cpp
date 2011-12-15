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

#include <cassert>

#include "tld/tld.h"

TldStruct tld;

int main(int argc, char* argv[]) {

	std::string videopath = "";
	int camindex = -1, startFrame = 0;
	Eigen::Vector4d initBB;
	initBB << -1, -1, -1, -1;
	unsigned int nodisplay = 0;
	CvCapture *captureSource;


	for (int i = 1; i < argc; ++i) {

		std::string current = argv[i];
		if (current == "-vid") {
			if (i + 1 <= argc && camindex == -1) {
				videopath = argv[i + 1];
				i++;
				continue;
			}
		} else if (current == "-cam") {
			if (i + 1 <= argc && videopath == "") {
				camindex = atoi(argv[i + 1]);
				i++;
				continue;
			}
		} else if (current == "-x") {
			if (i + 2 <= argc && initBB(0) == -1 && initBB(2) == -1) {
				initBB(0) = std::min(double(atoi(argv[i + 1])), double(atoi(
						argv[i + 2])));

				initBB(2) = std::max(double(atoi(argv[i + 1])), double(atoi(
						argv[i + 2])));
			}
		} else if (current == "-y") {
			if (i + 2 <= argc && initBB(1) == -1 && initBB(3) == -1) {
				initBB(1) = std::min(double(atoi(argv[i + 1])), double(atoi(
						argv[i + 2])));

				initBB(3) = std::max(double(atoi(argv[i + 1])), double(atoi(
						argv[i + 2])));
			}
		} else if (current == "-nodisplay") {
			nodisplay = 1;
		} else if (current == "-fr") {
			if (i + 1 <= argc) {
				startFrame = atoi(argv[i + 1]);
				i++;
				continue;
			}
		}
	}

	if (videopath == "" && camindex == -1) {
		exit(0);
	}

	if (initBB(0) == -1 || initBB(1) == -1 || initBB(2) == -1 || initBB(3)
			== -1)
		exit(0);

	tld.cfg.initBB = initBB;

	if(camindex >= 0)
		captureSource = cvCaptureFromCAM(camindex);
	else if(!videopath.empty())
		captureSource = cvCaptureFromAVI(videopath.c_str());
	else
		assert(false && "No image source.");

	tld.cfg.imgsource = new CvCaptureImageSource(captureSource);

	for(int i = 0; i < startFrame; i++) {
		// Ignore all frames up to startFrame.
		if(!tld.cfg.imgsource->nextImage())
			exit(0); // Ran out of images.
	}

	tld.cfg.plot.save = 0;
	tld.cfg.plot.patch_rescale = 1;
	tld.cfg.plot.pex = 0;
	tld.cfg.plot.nex = 0;
	tld.cfg.plot.target = 0;
	tld.cfg.plot.replace = 0;
	tld.cfg.plot.dt = 1;
	tld.cfg.plot.drawoutput = 3;
	tld.cfg.plot.confidence = 1;

	tld.model.num_trees = NTREES;
	tld.model.num_features = NFEATURES;

	Patchsize patchsize;
	patchsize.x = PATCHSIZE;
	patchsize.y = PATCHSIZE;
	tld.model.patchsize = patchsize;
	tld.model.min_win = 24;
	tld.model.fliplr = 0; // mirrored versions of object
	tld.model.ncc_thesame = 0.95;
	tld.model.valid = 0.5;
	tld.model.thr_fern = 0.5;
	tld.model.thr_nn = 0.65;
	tld.model.thr_nn_valid = 0.7;

	p_par p_par_init = { 10, 20, 5, 20, 0.02, 0.02 };
	p_par p_par_update = { 10, 10, 5, 10, 0.02, 0.02 };
	N_par n_par = { 0.2, 100 };
	Tracker tracker = { 10 };
	Control control = { 1, 1, 1, 1 };
	tld.p_par_init = p_par_init;
	tld.p_par_update = p_par_update;
	tld.n_par = n_par;
	tld.tracker = tracker;
	tld.control = control;

	tldExample(tld, !nodisplay);

}
