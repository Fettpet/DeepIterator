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

#define BOOST_TEST_MODULE NestedIterator
#include <boost/test/included/unit_test.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"

//using namespace boost::unit_test;
typedef int_fast32_t ParticleProperty;
typedef deepiterator::Particle<int_fast32_t, 2> Particle;
typedef deepiterator::Frame<Particle, 10> Frame;
typedef deepiterator::Supercell<Frame> Supercell;
typedef deepiterator::SupercellContainer<Supercell> SupercellContainer;


BOOST_AUTO_TEST_CASE(PositionsInFrames)
{

    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);
    
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    
/**
 * @brief The first test is used to verify the iteration over all particle 
 * attributes within a frame. 
 */

    auto && view = deepiterator::makeView(
                        *(supercell.firstFrame), 
                        deepiterator::makeIteratorPrescription(
                            deepiterator::makeAccessor(),
                            deepiterator::makeNavigator( 
                                Offset(0),
                                Jumpsize(1)
                            ),
                            deepiterator::makeIteratorPrescription(
                                deepiterator::makeAccessor(),
                                deepiterator::makeNavigator(
                                    Offset(0),
                                    Jumpsize(1)
                                )
                            )
                        )
                    );


    // 1. test ++it
    uint counter = 0u;
    for(auto && it = view.begin(); it != view.end(); ++it)
    {
        counter += (*it);
    }
    // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u);

    // 2. test it++
    counter = 0;
    for(auto && it = view.begin(); it != view.end(); it++)
    {
        counter += (*it);
    }
    // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u);



    // 3. test --it
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); ++it)
    {
        counter += (*it);
    }
    // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u);

    // 4. test it--
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it++)
    {
        counter += (*it);
    }

//     // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u); 

}


BOOST_AUTO_TEST_CASE(ParticleInSupercell)
{


    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);
    
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    
    auto && view = deepiterator::makeView( 
                               supercell,
                               deepiterator::makeIteratorPrescription(
                                    deepiterator::makeAccessor(),
                                    deepiterator::makeNavigator(
                                        Offset(0),
                                        Jumpsize(1)),
                                    deepiterator::makeIteratorPrescription(
                                        deepiterator::makeAccessor(),
                                        deepiterator::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)
                                        )
                                    )
                                )
    );


   // 1. test ++it
    uint counter = 0u;
    for(auto && it = view.begin(); it != view.end(); ++it)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);

    // 2. test it++
    counter = 0u;
    for(auto && it = view.begin(); it != view.end(); it++)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);

    // 3. test --it
    counter = 0u;


    for(auto && it = view.rbegin(); it != view.rend(); ++it)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);

    // 4. test it--
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it++)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);

}


BOOST_AUTO_TEST_CASE(PositionsInSupercell)
{

/**
 * @brief We use this test for 3 layer. 
 */

    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    
    
    auto && view = deepiterator::makeView(
        supercell, 
        makeIteratorPrescription(
            deepiterator::makeAccessor(),
            deepiterator::makeNavigator(
                Offset(0),
                Jumpsize(1)),
            deepiterator::makeIteratorPrescription(
                deepiterator::makeAccessor(),
                deepiterator::makeNavigator(
                    Offset(0),
                    Jumpsize(1)),
                deepiterator::makeIteratorPrescription(
                    deepiterator::makeAccessor(),
                    deepiterator::makeNavigator(
                        Offset(0),
                        Jumpsize(1))))));

    std::cout << "Data structure: " << std::endl << supercell << std::endl;
    // test ++it
    uint counter = 0u;
    for(auto it = view.begin(); it != view.end(); ++it)
    {

        counter++;            
    }

    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);
    
    // test it++
    counter = 0u;
    for(auto it = view.begin(); it != view.end(); it++)
    {
        counter++;            
    }

    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);
    
    //test --it
    counter = 0u;
    for(auto it = view.rbegin(); it != view.rend(); ++it)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);

    //test it--
    counter = 0u;
    for(auto it = view.rbegin(); it != view.rend(); it++)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);

}


