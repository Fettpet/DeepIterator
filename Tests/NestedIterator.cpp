
#define BOOST_TEST_MODULE NestedIterator
#include <boost/test/included/unit_test.hpp>

#include "PIC/SupercellContainer.hpp"
#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "DeepIterator.hpp"
#include "View.hpp"
#include "Definitions/hdinline.hpp"

//using namespace boost::unit_test;
typedef int_fast32_t ParticleProperty;
typedef hzdr::Particle<ParticleProperty, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::Supercell<Frame> Supercell;
typedef hzdr::SupercellContainer<Supercell> SupercellContainer;


BOOST_AUTO_TEST_CASE(PositionsInFrames)
{
 
    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);
    
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
/**
 * @brief The first test is used to verify the iteration over all particle 
 * attributes within a frame. 
 */
    auto && view = hzdr::makeView(
                        *(supercell.firstFrame), 
                        hzdr::makeIteratorPrescription(
                            hzdr::makeAccessor(),
                            hzdr::makeNavigator( 
                                Offset(0),
                                Jumpsize(1)),
                            hzdr::makeIteratorPrescription(
                                hzdr::makeAccessor(),
                                hzdr::makeNavigator(
                                    Offset(0),
                                    Jumpsize(1)))));
                            
    std::cout << *(supercell.firstFrame) << std::endl;
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
    for(auto && it = view.rbegin(); it != view.rend(); --it)
    {
        std::cout << *it << std::endl;
        counter += (*it);
    }
    // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u); 
    
    // 4. test it--
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it--)
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
    
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    auto && view = hzdr::makeView( 
                               supercell,
                               hzdr::makeIteratorPrescription(
                                    hzdr::makeAccessor(),
                                    hzdr::makeNavigator(
                                        Offset(0),
                                        Jumpsize(1)),
                                    hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)))));
     
    
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
    for(auto && it = view.rbegin(); it != view.rend(); --it)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);
    
    // 4. test it--
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it--)
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
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    
    auto && view = hzdr::makeView(
        supercell, 
        makeIteratorPrescription(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                Offset(0),
                Jumpsize(1)),
            hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(0),
                    Jumpsize(1)),
                hzdr::makeIteratorPrescription(
                    hzdr::makeAccessor(),
                    hzdr::makeNavigator(
                        Offset(0),
                        Jumpsize(1))))));
     
    std::cout << "Data structure: " << std::endl << supercell << std::endl;
    // test ++it
    uint counter = 0u;
    for(auto it = view.begin(); it != view.end(); ++it)
    {
        std::cout << *it << std::endl;
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
    for(auto it = view.rbegin(); it != view.rend(); --it)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);

    //test it--
    counter = 0u;
    for(auto it = view.rbegin(); it != view.rend(); it--)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);
}
