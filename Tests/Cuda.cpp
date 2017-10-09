
int main(){return 0;}
#if 0
/**
 * @author Sebastian Hahn < t.hahn@hzdr.de >
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

#include "Cuda/cuda.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "DeepIterator.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include "View.hpp"
#include "Definitions/hdinline.hpp"

typedef hzdr::Particle<int32_t, 2> Particle;
typedef hzdr::Frame<Particle, 256> Frame;
typedef hzdr::SuperCell<Frame> Supercell;

// 
// /**
//  * @brief Within this test, we touch all particles within a Frame 
//  */
BOOST_AUTO_TEST_CASE(PositionsInFrames)
{
    Supercell* super;
    auto nbParticleInLastFrame = 100;
    auto nbFrames = 5;
    callSupercellAddOne(&super, nbFrames, nbParticleInLastFrame);
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    auto && it = hzdr::makeIterator(*super, 
                             hzdr::makeAccessor(*super),
                             hzdr::makeNavigator(*super,
                                            hzdr::Direction::Forward(),
                                            Offset(0),
                                            Jumpsize(1)),
                             hzdr::make_child(hzdr::makeAccessor(),
                                        hzdr::makeNavigator(hzdr::Direction::Forward(),
                                                       Offset(0),
                                                       Jumpsize(1))));
                               

    auto counter=0;
    for(; not it.isAtEnd(); ++it)
    {
        counter++;
        BOOST_TEST((*it).data[0] == (*it).data[1]);
    }
    // 4 full Frames, 1 with 100 elements
    BOOST_TEST(counter == 256 * 4 +100);
}



BOOST_AUTO_TEST_CASE(AddAllParticlesInOne)
{

    typedef hzdr::SupercellContainer<Supercell> SupercellContainer;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    const int nbSupercells = 3;
    Supercell** super;
    std::vector<int> nbFrames, nbParticles;
    for(int i=0; i<nbSupercells; ++i)
    {
        nbFrames.push_back(rand()%16);
        nbParticles.push_back(rand()%256);
    }

    callSupercellSquareAdd(&super, nbSupercells, nbFrames, nbParticles);
    
    // all first elements need to have the same number of elements
    SupercellContainer supercellContainer(*super, nbSupercells);  
    

    
    auto it = hzdr::makeIterator(
                supercellContainer, 
                hzdr::makeAccessor(supercellContainer),
                hzdr::makeNavigator(
                    supercellContainer,
                    hzdr::Direction::Forward(),
                    Offset(0),
                    Jumpsize(1)));
    
    for(; not it.isAtEnd(); ++it)
    {
        auto itPart = hzdr::makeIterator(
            *it,
            hzdr::makeAccessor(*it),
            hzdr::makeNavigator(
                *it,
                hzdr::Direction::Forward(),
                Offset(0),
                Jumpsize(1)),
            hzdr::make_child(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    hzdr::Direction::Forward(),
                    Offset(0),
                    Jumpsize(1))));
        auto value = (*itPart).data[1];
        BOOST_TEST(value > 0);
        for(; not itPart.isAtEnd(); ++itPart)
        {
            BOOST_TEST((*itPart).data[1] == value);
        }
    }
    
}
#endif
