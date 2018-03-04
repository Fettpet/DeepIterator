
/**
 * @author Sebastian Hahn t.hahn <at> hzdr.de
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

typedef hzdr::Particle<int_fast32_t, 2u> Particle;
typedef hzdr::Frame<Particle, 10u> Frame;
typedef hzdr::Supercell<Frame> Supercell;
typedef hzdr::SupercellContainer<Supercell> SupercellContainer;

BOOST_AUTO_TEST_CASE(Frames)
{
 
    Frame testFrame;
    
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    std::array<Particle, 10u> buffer;
    // 0. We create a concept
    auto && concept = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0u),
            Jumpsize(1u)));
    auto && view = hzdr::makeView(testFrame, concept);
    
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

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    auto && view = hzdr::makeView(
                        supercell, 
                        makeIteratorPrescription(
                            hzdr::makeAccessor(),
                            hzdr::makeNavigator(
                                Offset(0u),
                                Jumpsize(1u)),
                            hzdr::makeIteratorPrescription(
                                hzdr::makeAccessor(),
                                hzdr::makeNavigator(
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
