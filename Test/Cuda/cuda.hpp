/* Copyright 2018 Sebastian Hahn

 * This file is part of DeepIterator.
 *
 * DeepIterator is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DeepIterator is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DeepIterator.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once 


#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"


typedef deepiterator::Particle<int32_t, 2> Particle;
typedef deepiterator::Frame<Particle, 256> Frame;
typedef deepiterator::Supercell<Frame> Supercell;

/**
 * @brief within this test, we add 1 to the first variable of each particle. 
 * This means, that the values of both variables in each particle are equal.
 * @param supercell out: the resulting supercell, will be create with a new. 
 * So keep sure to delete it.
 * @param Frame in: number of Frames within the Supercell,
 * @param nbParticleInFrame in: number of Particles within the last Frame
 */
void
callSupercellAddOne(Supercell** supercell, int Frames, int nbParticleInFrame);


/**
 * @brief We use this test to add all values of a particle to another supercell. Example with 2 supercells
 * foreach(Particle p1 in Supercell[0])
 *  foreach(Particle p2 in Supercell[1])
 *      p1.value += p2.value;
 *  
 */
void callSupercellSquareAdd(Supercell*** superCellContainer, int nbSupercells, std::vector<int> nbFramesSupercell, std::vector<int> nbParticlesInFrame);

