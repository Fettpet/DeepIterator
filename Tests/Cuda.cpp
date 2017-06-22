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
#include "View.hpp"
#include "Definitions/hdinline.hpp"
#include "Iterator/RuntimeTuple.hpp"

typedef hzdr::Particle<int32_t, 2> Particle;
typedef hzdr::Frame<Particle, 256> Frame;
typedef hzdr::SuperCell<Frame> Supercell;


/**
 * @brief Within this test, we touch all particles within a Frame 
 */
BOOST_AUTO_TEST_CASE(PositionsInFrames)
{
    Supercell* super;
    auto nbParticleInLastFrame = 100;
    auto nbFrames = 5;
    callSupercellAddOne(&super, nbFrames, nbParticleInLastFrame);
    
    const int jumpsizeParticle = 1;
    const int offsetParticle = 0;
    const int nbElementsParticle = nbParticleInLastFrame;
    typedef hzdr::runtime::TupleFull RuntimeTuple;
    
    const RuntimeTuple runtimeVarParticle(offsetParticle, nbElementsParticle, jumpsizeParticle);
    
    
    const int jumpsizeFrame = 1;
    const int offsetFrame = 0;
    const int nbElementsFrame = 0;
    const RuntimeTuple runtimeFrame(offsetFrame, nbElementsFrame, jumpsizeFrame);
    
    typedef hzdr::View<Frame, hzdr::Direction::Forward,  hzdr::Collectivity::None,RuntimeTuple> ParticleInFrame;
    
    hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::CudaIndexable, RuntimeTuple, ParticleInFrame> view(super, runtimeFrame, ParticleInFrame(nullptr, runtimeVarParticle)); 
    
    auto counter=0;
    auto it=view.begin();
    for(; it!=view.end(); ++it)
    {
        if(*it)
        {
            counter++;
            BOOST_TEST((**it).data[0] == (**it).data[1]);
        }
    }
    // 4 full Frames, 1 with 100 elements
    BOOST_TEST(counter == 256 * 4 +100);
}



BOOST_AUTO_TEST_CASE(AddAllParticlesInOne)
{
    
    Supercell** super;
    std::vector<int> nbFrames{2,3,1};
    std::vector<int> nbParticles{100,150,100};
    callSupercellSquareAdd(&super, 3, nbFrames, nbParticles);
    
    
    // I have supercells, and I need 
    std::cout <<"Superzelle 1" << std::endl;
    std::cout << *(super[0]);
    
    std::cout <<"Superzelle 2" << std::endl;
    std::cout << *(super[1]);
    
    std::cout <<"Superzelle 3" << std::endl;
     std::cout << *(super[2]);
}
