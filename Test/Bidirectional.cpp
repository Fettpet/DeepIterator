
/**
 * @author Sebastian Hahn t.hahn <at> hzdr.de
 * @brief We verify the correct implementation of the bidirectional functionality.
 * We need to test two operations:
 * 1. --it
 * 2. it--
 * What we also need to test:
 * -- rbegin and rend
 * -- The iteration in two differnt directions
 * We need to test different offset and jumpsizes
 */

#define BOOST_TEST_MODULE Bidirectional
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
    for(auto && it = view.rbegin(); it != view.rend(); --it)
    {
        std::cout << *it <<std::endl;
        buffer[counter++] = *it;
    }
    BOOST_TEST(counter == 10u);
    
    // 2. Test the read results
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it--)
    {
        BOOST_TEST(buffer[counter++] == (*it));
    }
    BOOST_TEST(counter == 10u);
    
    // 3. write new particles in the frame
    Particle particle = Particle(0,0);
    
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it--)
    {
        *it = particle;
    }
    
    // test the result
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it--)
    {
        BOOST_TEST((*it) == particle);
    }
    
    auto && it = view.begin();
    
    BOOST_TEST((it == --(++it)));
    
    // test the bidirectional iterator property
    BOOST_TEST((--(view.begin()) == view.rend()));
    BOOST_TEST((view.end() == ++(view.rbegin())));
    
    BOOST_TEST(( ++(--view.begin()) == view.begin()));
    BOOST_TEST(( --(++view.rbegin()) == view.rbegin()));
}


BOOST_AUTO_TEST_CASE(FramesDiffentOffsetJumpsizes)
{
    


    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;

        
    std::vector<uint_fast32_t> offsets({0u, 1u, 2u, 3u, 4u});
    std::vector<uint_fast32_t> jumpsizes({1u, 2u, 3u, 4u});
    
    const static uint nbParticle = 10u;
    std::array<Particle, nbParticle> buffer;
    
    
    for( auto off: offsets)
    {
        for(auto jump: jumpsizes)
        {   

            Frame testFrame;
            // 0. We create a concept
            auto && concept = hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(off),
                    Jumpsize(jump)));
            auto && view = hzdr::makeView(testFrame, concept);
            
            // 1. Test we iterate over the frame and read all particles 
            auto counter = 0u;
            for(auto && it = view.rbegin(); it != view.rend(); --it)
            {
                if(off == 1 and jump == 2)
                    std::cout << *it << std::endl;
                buffer[counter++] = *it;
            }
            BOOST_TEST(counter == (nbParticle - off+ jump - 1) / jump);
            
            // 2. Test the read results
            counter = 0u;
            for(auto && it = view.rbegin(); it != view.rend(); it--)
            {
                BOOST_TEST(buffer[counter++] == (*it));
            }
            BOOST_TEST(counter == (nbParticle - off + jump - 1) / jump);
            
            // 3. write new particles in the frame
            Particle particle = Particle(0,0);
            
            for(auto && it = view.rbegin(); it != view.rend(); it--)
            {
                *it = particle;
            }
            
            // test the result
            for(auto && it = view.rbegin(); it != view.rend(); it--)
            {
                BOOST_TEST((*it) == particle);
            }
            
            auto && it = view.begin();
            
            BOOST_TEST((it == --(++it)));
            
            // test the bidirectional iterator property
            BOOST_TEST((--(view.begin()) == view.rend()));
            BOOST_TEST((view.end() == ++(view.rbegin())));
            
            BOOST_TEST(( ++(--view.begin()) == view.begin()));
            BOOST_TEST(( --(++view.rbegin()) == view.rbegin()));
        }   
    } 
}


BOOST_AUTO_TEST_CASE(ParticleInSupercell)
{
    uint_fast32_t nbFrames = 5u;
    uint_fast32_t nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    
    auto && concept = makeIteratorPrescription(
                            hzdr::makeAccessor(),
                            hzdr::makeNavigator(
                                Offset(0u),
                                Jumpsize(1u)),
                            hzdr::makeIteratorPrescription(
                                hzdr::makeAccessor(),
                                hzdr::makeNavigator(
                                    Offset(0u),
                                    Jumpsize(1u))));
    auto && view = hzdr::makeView(
                        supercell,
                        concept);
     
    

    uint counter{0u};
    for(auto it=view.rbegin(); it != view.rend(); --it)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);
    
    counter = 0u;
    for(auto it=view.rbegin(); it != view.rend(); it--)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);

    // check the border properties of a bidirectional iterator
    BOOST_TEST((--(view.begin()) == view.rend()));
    BOOST_TEST((view.end() == ++(view.rbegin())));
    
    BOOST_TEST(( ++(--view.begin()) == view.begin()));
    BOOST_TEST(( --(++view.rbegin()) == view.rbegin()));
}

BOOST_AUTO_TEST_CASE(Borders)
{
    uint_fast32_t nbFrames = 2u;
    uint_fast32_t nbParticlesInLastFrame = 1u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    std::vector<uint_fast32_t> offsetsInner({0u, 1u});
    std::vector<uint_fast32_t> jumpsizesInner({1u, 2u, 3u, 4u});
    std::cout << supercell << std::endl;
    for(auto jump: jumpsizesInner)
        for(auto off: offsetsInner)
        {
            std::cout << "(" << jump << ", " << off << ")" << std::endl;
            auto && view = hzdr::makeView(
                        supercell, 
                        hzdr::makeIteratorPrescription(
                            hzdr::makeAccessor(),
                            hzdr::makeNavigator( 
                                Offset(0u),
                                Jumpsize(1u)),
                            hzdr::makeIteratorPrescription(
                                hzdr::makeAccessor(),
                                hzdr::makeNavigator(
                                    Offset(3u),
                                    Jumpsize(1u)),
                                hzdr::makeIteratorPrescription(
                                    hzdr::makeAccessor(),
                                    hzdr::makeNavigator(
                                        Offset(off),
                                        Jumpsize(jump))))));

            auto sumForward = 0u;
            for(auto && it = view.begin(); it != view.end(); ++it)
            {

                sumForward += *it;
            }
            
            auto sumBackward = 0u;
           
            for(auto && it = view.rbegin(); it != view.rend(); --it)
            {

                sumBackward += *it;
            }
            BOOST_TEST(sumForward == sumBackward);
        }
    

}



BOOST_AUTO_TEST_CASE(ParticleInSupercellDifferentOffsets)
{
    uint_fast32_t nbFrames = 5u;
    uint_fast32_t nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    std::vector<uint_fast32_t> offsetsInner({0u, 1u, 2u, 3u});
    std::vector<uint_fast32_t> jumpsizesInner({1u, 2u, 3u, 4u});
    std::cout << supercell << std::endl;
    for(auto off: offsetsInner)
        for(auto jump: jumpsizesInner)
        {
            auto && concept = makeIteratorPrescription(
                                    hzdr::makeAccessor(),
                                    hzdr::makeNavigator(
                                        Offset(off),
                                        Jumpsize(jump)),
                                    hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0u),
                                            Jumpsize(1u))));
            auto && view = hzdr::makeView(
                                supercell,
                                concept);
            
            // calc the number of elements
            // fullframe
            auto nbParticles = 0u;
            for(auto i =off; i<nbFrames-1u; i+=jump)
            {
                nbParticles += 10;
            }
            // if the last frame is hit
            // minus 1u since we start at 0 with counting
            nbParticles += ((nbFrames - 1u - off) % jump == 0 and off < nbFrames) * nbParticlesInLastFrame;

            uint counter{0u};
            for(auto it=view.rbegin(); it != view.rend(); --it)
            {
                if(off == 0u and jump == 3u)
                    std::cout << *it << std::endl;
                counter++;            
            }
            
            // There are 4 full frames with 10 Elements an one frame with 2 elements
            BOOST_TEST(counter == nbParticles);
            
            counter = 0u;
            for(auto it=view.rbegin(); it != view.rend(); it--)
            {
                counter++;            
            }
            // There are 4 full frames with 10 Elements an one frame with 2 elements
            BOOST_TEST(counter == nbParticles);
            
            /*******************
             * Second test: We test the inner jumpsize and offset
             * 
             * *****************/
            auto && conceptInner = makeIteratorPrescription(
                                    hzdr::makeAccessor(),
                                    hzdr::makeNavigator(
                                        Offset(0u),
                                        Jumpsize(1u)),
                                    hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jump))));
            auto && viewInner = hzdr::makeView(
                                supercell,
                                conceptInner);
            
            // calc the number of elements
            // fullframe
            nbParticles = 0u;
            for(auto i =off; i<10u; i+=jump)
            {
                ++nbParticles;

            }
            nbParticles *= nbFrames - 1u;
            // add the last frame
            for(auto j=off; j < nbParticlesInLastFrame; j+=jump)
            {
                ++nbParticles;
            }
            counter = 0u;
            for(auto it=viewInner.rbegin(); it != viewInner.rend(); --it)
            {
                std::cout << *it << std::endl;
                counter++;            
            }
            
            // There are 4 full frames with 10 Elements an one frame with 2 elements
            
            BOOST_TEST(counter == nbParticles);
            
            counter = 0u;
            for(auto it=viewInner.rbegin(); it != viewInner.rend(); it--)
            {
                counter++;            
            }
            // There are 4 full frames with 10 Elements an one frame with 2 elements
            BOOST_TEST(counter == nbParticles);
        }

}

