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
 * @author Sebastian Hahn
 * @brief In this test collection we verify the functionality of the simple 
 * forward iteration. We use the PIConGPU data structures for our tests. 
 * 1. Test Frames: Read all particles in a frame and write particles in the frame
 * 2. Test Supercells: Read all particles in a supercell
 * 
 */

#define BOOST_TEST_MODULE ForwardIterator
#include <boost/test/included/unit_test.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"

using namespace boost::unit_test;

typedef deepiterator::Particle<int_fast32_t, 2u> Particle;
typedef deepiterator::Frame<Particle, 10u> Frame;
typedef deepiterator::Supercell<Frame> Supercell;
typedef deepiterator::SupercellContainer<Supercell> SupercellContainer;

BOOST_AUTO_TEST_CASE(Frames)
{
 
    Frame testFrame;
    
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    std::array<Particle, 10u> buffer;
    // 0. We create a concept
    auto && concept = deepiterator::makeIteratorPrescription(
        deepiterator::makeAccessor(),
        deepiterator::makeNavigator(
            Offset(0u),
            Jumpsize(1u)));
    auto && view = deepiterator::makeView(testFrame, concept);
    
    // 1. Test we iterate over the frame and read all particles 
    auto counter = 0u;
    for(auto && it = view.begin(); it != view.end(); ++it)
    {
        buffer[counter++] = *it;
    }
    BOOST_TEST(counter == 10u);
    
    // 2. Test the read results
    counter = 0u;
    for(auto && it = view.begin(); it != view.end(); it++)
    {
        BOOST_TEST(buffer[counter++] == (*it));
    }
    BOOST_TEST(counter == 10u);
    
    // 3. write new particles in the frame
    Particle particle = Particle(0,0);
    
    counter = 0u;
    for(auto && it = view.begin(); it != view.end(); it++)
    {
        *it = particle;
    }
    
    // test the result
    counter = 0u;
    for(auto && it = view.begin(); it != view.end(); it++)
    {
        BOOST_TEST((*it) == particle);
    }
}




BOOST_AUTO_TEST_CASE(ParticleInSupercell)
{

    uint_fast32_t nbFrames = 5u;
    uint_fast32_t nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);

    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    auto && view = deepiterator::makeView(
                        supercell, 
                        makeIteratorPrescription(
                            deepiterator::makeAccessor(),
                            deepiterator::makeNavigator(
                                Offset(0u),
                                Jumpsize(1u)),
                            deepiterator::makeIteratorPrescription(
                                deepiterator::makeAccessor(),
                                deepiterator::makeNavigator(
                                    Offset(0u),
                                    Jumpsize(1u)))));
     
    

    uint counter{0u};
    for(auto it=view.begin(); it != view.end(); ++it)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);

}
