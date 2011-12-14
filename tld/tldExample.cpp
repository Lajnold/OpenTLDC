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

// Instance of tracking learning detection structures
TldStruct tld;

/**
 *  Loop function of TLDC.
 *
 *  @param opt tld structure with initial values and thresholds
 *  @param cfg stream settings and initial bounding box
 */
void tldExample(TldStruct* opt, Config& cfg) {

	srand(0);

	double t = (double)getTickCount();

	tld = *opt;
	tld.cfg = &cfg;

	/* INITIALIZATION -------------------------------------- */

	if(!tldInit(tld))
		return; // No images.

	if (!cfg.nodisplay)
		tldDisplay(0, 0, tld, t);

	/* RUN-TIME -------------------------------------------- */

	unsigned long i = 1;

	while (i < 2500) {

		t = (double)getTickCount();

		if(!tldProcessFrame(tld, i))
			break; // Out of images.

		t = ((double)getTickCount() - t)/getTickFrequency();

		if (!cfg.nodisplay)
			tldDisplay(1, i, tld, t);

		std::cout << "BB - xmin: " << tld.currentBB(0) << " ymin: "
				<< tld.currentBB(1) << " xmax: " << tld.currentBB(2) << " ymax: "
				<< tld.currentBB(3) << std::endl;

		i++;

	}

	if(!cfg.nodisplay)
		cvDestroyWindow("Result");
}
