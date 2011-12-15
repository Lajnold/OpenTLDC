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

#ifndef TLD_H_
#define TLD_H_

#include "structs.h"

/* Shows Results in additional window */
void tldDisplay(int i, unsigned long index, TldStruct& tld, double fps);

/* Main Loop */
void tldExample(TldStruct& tld, bool display);

/* measures initial structures */
bool tldInit(TldStruct& tld/*, CamImage& source, Person& persondetect*/);

/* main method, is called on each loop */
bool tldProcessFrame(TldStruct& tld, unsigned long i);

/** Sets a new bounding box for the object. */
void tldSetBB(TldStruct& tld, Eigen::Vector4d& bb);

#endif /* TLD_H_ */

