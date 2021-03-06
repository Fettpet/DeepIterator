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

/**
 * @author Sebastian Hahn <  >
 * @brief Within this file we test the cuda implementation of the DeepIterator. 
 * Because there are problems with BOOST::TEST and Cuda, we need three files to test 
 * the CUDA Implementation:
 * 1. Cuda.cpp: This is the main file. Here we check the results of the kernel calls 
 * 2. Cuda.hpp: We define the Header of the test here.
 * 3. Cuda.cu: The test on GPU are here defined. 
 * 
 */
#define BOOST_MPL_CFG_GPU_ENABLED
#define BOOST_TEST_MODULE CudaIterator
#include <boost/test/included/unit_test.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"
#include "deepiterator/../../Test/Cuda/cuda.hpp"
typedef deepiterator::Particle<int32_t, 2> Particle;
typedef deepiterator::Frame<Particle, 256> Frame;
typedef deepiterator::Supercell<Frame> Supercell;

// 
// /**
//  * @brief Within this test, we touch all particles within a Frame 
//  */
BOOST_AUTO_TEST_CASE(PositionsInFrames)
{
    Supercell* supercell;
    auto nbParticleInLastFrame = 100u;
    auto nbFrames = 5u;
    callSupercellAddOne(&supercell, nbFrames, nbParticleInLastFrame);
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    
    auto concept = deepiterator::makeIteratorPrescription(
                            deepiterator::makeAccessor(),
                            deepiterator::makeNavigator(
                                Offset(0u),
                                Jumpsize(1u)),
                            deepiterator::makeIteratorPrescription(
                                deepiterator::makeAccessor(),
                                deepiterator::makeNavigator(
                                    Offset(0u),
                                    Jumpsize(1u))));
    
    auto view = deepiterator::makeView(*supercell, concept);

                               

    auto counter = 0u;
    for(auto it=view.begin(); it != view.end(); ++it)
    {
        counter++;
        BOOST_TEST((*it).data[0u] == (*it).data[1u]);
    }
    // 4 full Frames, 1 with 100 elements
    BOOST_TEST(counter == 256u * 4u + 100u);
}



BOOST_AUTO_TEST_CASE(AddAllParticlesInOne)
{

    typedef deepiterator::SupercellContainer<Supercell> SupercellContainer;
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    
    const uint nbSupercells = 3u;
    Supercell** super;
    std::vector<int> nbFrames, nbParticles;
    for(uint i=0u; i<nbSupercells; ++i)
    {
        nbFrames.push_back(rand()%16u);
        nbParticles.push_back(rand()%256u);
    }

  //  callSupercellSquareAdd(&super, nbSupercells, nbFrames, nbParticles);
    
    // all first elements need to have the same number of elements
    SupercellContainer supercellContainer(
        *super,
         nbSupercells
    );  
    
    auto concept = deepiterator::makeIteratorPrescription(
        deepiterator::makeAccessor(),
        deepiterator::makeNavigator(
            Offset(0u),
            Jumpsize(1u)));
    std::cout << supercellContainer[2] << std::endl;                         
 
     auto view = deepiterator::makeView(supercellContainer, concept);
 
     for(auto it=view.begin(); it!=view.end(); ++it)
     {
         auto conceptParticle = deepiterator::makeIteratorPrescription(
             deepiterator::makeAccessor(),
             deepiterator::makeNavigator(
                 Offset(0u),
                 Jumpsize(1u)),
             deepiterator::makeIteratorPrescription(
                 deepiterator::makeAccessor(),
                 deepiterator::makeNavigator(
                     Offset(0u),
                     Jumpsize(1u))));
         auto viewParticle = deepiterator::makeView(*it, conceptParticle);
         auto itParticle = viewParticle.begin();
         auto value = (*itParticle).data[0u];
         BOOST_TEST((value > 0));
         for(; itParticle != viewParticle.end(); ++itParticle)
         {
             BOOST_TEST((*itParticle).data[1u] == value);
         }
     }
    
}

