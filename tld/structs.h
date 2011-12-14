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

#define NFEATURES 13 // Features per tree
#define NTREES 10 // number of trees
#define PATCHSIZE 15 // 15 x 15
#define MAXPATCHES 100
#define MAXDT 100
#define DIMX 640
#define DIMY 480

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <Eigen/Dense>
#include <Eigen/Core>

// Initial Configuration
typedef struct {
	Eigen::Vector4d init;
	int camindex;
	unsigned int nodisplay;
	int startFrame;
	std::string videopath;
} Config;

// Plot
typedef struct {
	unsigned int save;
	unsigned int pex;
	unsigned int nex;
	unsigned int target;
	unsigned int replace;
	unsigned int dt;
	unsigned int confidence;
	unsigned int drawoutput;
	double patch_rescale;
} Plot;

// Blured image and input image (grayscale)
typedef struct {
	IplImage* blur;
	IplImage* input;
} ImgType;

typedef struct {
	unsigned int m; // rows
	unsigned int n; // cols
} ImgSize;

typedef struct {
	unsigned int x;
	unsigned int y;
} Patchsize;

// Carries initial thresholds
typedef struct {
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
	unsigned int num_init;
} Model;

// Temporal confidelity and pattern
typedef struct {
	Eigen::VectorXd conf;
	Eigen::Matrix<double, 10, Eigen::Dynamic> patt;
} Tmp;

typedef struct {
	unsigned int num_closest;
	unsigned int num_warps;
	unsigned int noise;
	unsigned int angle;
	double shift;
	double scale;
} p_par;

typedef struct {
	double overlap;
	unsigned int num_patches;
} N_par;

typedef struct {
	unsigned int occlusion;
} Tracker;

typedef struct {
	unsigned char maxbbox;
	unsigned char update_detector;
	unsigned char drop_img;
	unsigned char repeat;
} Control;

typedef struct {
	Eigen::Matrix<double, 4, MAXDT> bb;
	int nbb;
	Eigen::Matrix<double, MAXDT, 1> conf2;
	Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), MAXDT> patch;
} Detection;

// Structure of TLD
typedef struct {
	Plot* plot;
	Model* model;
	Config* cfg;
	Tmp tmp;
	p_par p_par_init;
	p_par p_par_update;
	N_par n_par;
	Tracker tracker;
	Control control;

	Eigen::Matrix<double, 4 * NFEATURES, NTREES> features;
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
	double size;

	ImgSize imgsize;
	IplImage* target;

	int npex;
	int nnex;

	Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), MAXPATCHES> pex;
	Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), MAXPATCHES> nex;

	//Eigen::MatrixXd xFJ;

	double var;

	IplImage* handle;
} TldStruct;

#endif
