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

#ifndef STRUCTS_H_
#define STRUCTS_H_

#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "tldconst.h"
#include "../img/ImageSource.h"

// Plot
struct Plot {
	//unsigned int save;
	unsigned int pex;
	unsigned int nex;
	unsigned int target;
	unsigned int replace;
	//unsigned int dt;
	//unsigned int confidence;
	unsigned int drawoutput;
	double patch_rescale;
};

// Configuration
struct Config {
	ImageSource *imgsource;
	Eigen::Vector4d newBB;
	bool newBBSet;
	Plot plot;

	Config() : newBBSet(false) { }
};


// Blured image and input image (grayscale)
struct ImgType {
	CvImage blur;
	CvImage input;
};

struct Patchsize {
	unsigned int x;
	unsigned int y;
};

// Carries initial thresholds
struct Model {
	Patchsize patchsize;
	unsigned char fliplr;
	unsigned int min_win;
	unsigned char num_trees;
	unsigned char num_features;
	double ncc_thesame;
	double valid;
	double thr_fern;
	double thr_nn;
	double thr_nn_valid;
	//unsigned int num_init;
};

// Temporal confidelity and pattern
struct Tmp {
	Eigen::VectorXd conf;
	Eigen::Matrix<double, 10, Eigen::Dynamic> patt;
};

struct P_par {
	unsigned int num_closest;
	unsigned int num_warps;
	unsigned int noise;
	unsigned int angle;
	double shift;
	double scale;
};

struct N_par {
	double overlap;
	unsigned int num_patches;
};

struct Tracker {
	unsigned int occlusion;
};

struct Control {
	unsigned char maxbbox;
	unsigned char update_detector;
	unsigned char drop_img;
	unsigned char repeat;
};

struct Detection {
	Eigen::Matrix<double, 4, TLD_MAXDT> bb;
	int nbb;
	Eigen::Matrix<double, TLD_MAXDT, 1> conf2;
	Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), TLD_MAXDT> patch;
};

// Structure of TLD
struct TldStruct {
	Model model;
	Config cfg;
	Tmp tmp;
	P_par p_par_init;
	P_par p_par_update;
	N_par n_par;
	Tracker tracker;
	Control control;

	Eigen::Matrix<double, 4 * TLD_NFEATURES, TLD_NTREES> features;
	int nGrid;

	Eigen::Matrix<double, 6, Eigen::Dynamic> grid;
	Eigen::Matrix<double, 2, 21> scales;

	ImgType prevImg;
	ImgType currentImg;
	Detection dt;
	Eigen::Vector4d prevBB;
	Eigen::Vector4d currentBB;
	double conf;
	double prevValid;
	double currentValid;
	//double size;

	CvImage target;

	int npex;
	int nnex;

	Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), TLD_MAXPATCHES> pex;
	Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), TLD_MAXPATCHES> nex;

	//Eigen::MatrixXd xFJ;

	double var;

	CvImage handle;
};

#endif
