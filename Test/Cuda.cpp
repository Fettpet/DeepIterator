
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
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"
#include "deepiterator/../../Test/Cuda/cuda.hpp"
typedef hzdr::Particle<int32_t, 2> Particle;
typedef hzdr::Frame<Particle, 256> Frame;
typedef hzdr::Supercell<Frame> Supercell;

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
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    auto concept = hzdr::makeIteratorPrescription(
                            hzdr::makeAccessor(),
                            hzdr::makeNavigator(
                                Offset(0u),
                                Jumpsize(1u)),
                            hzdr::makeIteratorPrescription(
                                hzdr::makeAccessor(),
                                hzdr::makeNavigator(
                                    Offset(0u),
                                    Jumpsize(1u))));
    
    auto view = hzdr::makeView(*supercell, concept);

                               

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

    typedef hzdr::SupercellContainer<Supercell> SupercellContainer;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
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
    
    auto concept = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0u),
            Jumpsize(1u)));
    std::cout << supercellContainer[2] << std::endl;                         
 
     auto view = hzdr::makeView(supercellContainer, concept);
 
     for(auto it=view.begin(); it!=view.end(); ++it)
     {
         auto conceptParticle = hzdr::makeIteratorPrescription(
             hzdr::makeAccessor(),
             hzdr::makeNavigator(
                 Offset(0u),
                 Jumpsize(1u)),
             hzdr::makeIteratorPrescription(
                 hzdr::makeAccessor(),
                 hzdr::makeNavigator(
                     Offset(0u),
                     Jumpsize(1u))));
         auto viewParticle = hzdr::makeView(*it, conceptParticle);
         auto itParticle = viewParticle.begin();
         auto value = (*itParticle).data[0u];
         BOOST_TEST((value > 0));
         for(; itParticle != viewParticle.end(); ++itParticle)
         {
             BOOST_TEST((*itParticle).data[1u] == value);
         }
     }
    
}

