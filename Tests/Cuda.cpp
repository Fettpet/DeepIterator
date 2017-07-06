
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
    


    
    
    
    typedef hzdr::View<Frame, hzdr::Direction::Forward,  hzdr::Collectivity::None> ParticleInFrame;
    
    hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::CudaIndexable, ParticleInFrame> view(super); 
    
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
    typedef hzdr::View<Frame, 
                       hzdr::Direction::Forward,  
                       hzdr::Collectivity::None> ParticleInFrame;
    typedef  hzdr::View<Supercell,
                        hzdr::Direction::Forward,  
                        hzdr::Collectivity::CudaIndexable, 
                        ParticleInFrame> FrameInSupercellView;
    typedef hzdr::SupercellContainer<Supercell> SupercellContainer;
    typedef hzdr::View<SupercellContainer, 
                       hzdr::Direction::Forward, 
                       hzdr::Collectivity::None> ViewSupercellContainer;
    const int nbSupercells = 3;
    Supercell** super;
    std::vector<int> nbFrames, nbParticles;
    for(int i=0; i<nbSupercells; ++i)
    {
        nbFrames.push_back(4);
        nbParticles.push_back(rand()%256);
    }

    callSupercellSquareAdd(&super, nbSupercells, nbFrames, nbParticles);
    
    // all first elements need to have the same number of elements
    SupercellContainer supercellContainer(*super, nbSupercells);  
    
    

    ViewSupercellContainer viewSupercellContainer(supercellContainer);
    
    for(auto it=viewSupercellContainer.begin();
        it != viewSupercellContainer.end();
        ++it)
    {
    
        
        FrameInSupercellView view(**it);
        const auto value = (**(view.begin())).data[1];
        
        for(auto itElem=view.begin(); itElem != view.end(); ++itElem)
        {
            if(*itElem)
            {
                BOOST_TEST( (**itElem).data[1] == value);
            }
        }
    }
//     
// 
//     
//     
//     // I have supercells, and I need 
// //     std::cout <<"Superzelle 1" << std::endl;
// //     std::cout << *(super[0]);
// //     
// //     std::cout <<"Superzelle 2" << std::endl;
// //     std::cout << *(super[1]);
// //     
// //     std::cout <<"Superzelle 3" << std::endl;
// //      std::cout << *(super[2]);
}
