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

TldStruct opt;
Plot plot;
Model model;
Config cfg;

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

	cfg.init = initBB;
	cfg.nodisplay = nodisplay;
	//cfg.startFrame = startFrame;

	if(camindex >= 0)
		captureSource = cvCaptureFromCAM(camindex);
	else if(!videopath.empty())
		captureSource = cvCaptureFromAVI(videopath.c_str());
	else
		assert(false && "No image source.");

	cfg.imgsource = new CvCaptureImageSource(captureSource);

	for(int i = 0; i < startFrame; i++) {
		// Ignore all frames up to startFrame.
		if(!cfg.imgsource->nextImage())
			exit(0); // Ran out of images.
	}

	opt.plot = &plot;
	opt.plot->save = 0;
	opt.plot->patch_rescale = 1;
	opt.plot->pex = 1;
	opt.plot->nex = 1;
	opt.plot->target = 0;
	opt.plot->replace = 0;
	opt.plot->dt = 1;
	opt.plot->drawoutput = 3;
	opt.plot->confidence = 1;

	opt.model = &model;
	opt.model->num_trees = NTREES;
	opt.model->num_features = NFEATURES;

	Patchsize patchsize;
	patchsize.x = PATCHSIZE;
	patchsize.y = PATCHSIZE;
	opt.model->patchsize = patchsize;
	opt.model->min_win = 24;
	opt.model->fliplr = 0; // mirrored versions of object
	opt.model->ncc_thesame = 0.95;
	opt.model->valid = 0.5;
	opt.model->thr_fern = 0.5;
	opt.model->thr_nn = 0.65;
	opt.model->thr_nn_valid = 0.7;

	p_par p_par_init = { 10, 20, 5, 20, 0.02, 0.02 };
	p_par p_par_update = { 10, 10, 5, 10, 0.02, 0.02 };
	N_par n_par = { 0.2, 100 };
	Tracker tracker = { 10 };
	Control control = { 1, 1, 1, 1 };
	opt.p_par_init = p_par_init;
	opt.p_par_update = p_par_update;
	opt.n_par = n_par;
	opt.tracker = tracker;
	opt.control = control;

	tldExample(&opt, cfg);

}
