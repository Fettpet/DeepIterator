#define BOOST_TEST_MODULE UnnestedIterator
#include <boost/test/included/unit_test.hpp>
#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "DeepIterator.hpp"
#include "DeepView.hpp"
#include "Iterator/RuntimeTuple.hpp"
using namespace boost::unit_test;
typedef hzdr::Particle<int, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::SuperCell<Frame> Supercell;
    typedef hzdr::RuntimeTuple<hzdr::runtime::offset::enabled,
                               hzdr::runtime::nbElements::enabled,
                               hzdr::runtime::jumpsize::enabled> RuntimeTuple;
BOOST_AUTO_TEST_CASE(PositionsInFrames)
{
 
    Supercell cell(5, 2);
    
    const int jumpsizeParticle = 1;
    const int offsetParticle = 0;
    const int nbElementsParticle = 2;
    
    const RuntimeTuple runtimeVarParticle(jumpsizeParticle, nbElementsParticle, offsetParticle);
    
    const int jumpsizeFrame = 1;
    const int offsetFrame = 0;
    const int nbElementsFrame = 0;
    
    const RuntimeTuple runtimeFrame(jumpsizeFrame, nbElementsFrame, offsetFrame);
    
    typedef hzdr::View<Particle, hzdr::Direction::Forward, hzdr::Collectivity::NonCollectiv, RuntimeTuple> PositionInParticleContainer;
    
    hzdr::View< Frame, 
                    hzdr::Direction::Forward, 
                    
                    hzdr::Collectivity::NonCollectiv, 
                    RuntimeTuple,
                    PositionInParticleContainer > test(cell.firstFrame, 
                                                       runtimeFrame,
                                                       PositionInParticleContainer(nullptr, runtimeVarParticle)); 
                     
    uint counter=0;
    for(auto it=test.begin(); it!=test.end(); ++it)
    {
        if(*it)
            ++counter;
    }
    
    // 10 Particles with 2 variables
    BOOST_TEST(counter == 20); 
 
}



BOOST_AUTO_TEST_CASE(ParticleInSuperCell)
{



    

    Supercell cell(5, 2);
    
    const int jumpsizeParticle = 1;
    const int offsetParticle = 0;
    const int nbElementsParticle = 2;
    
    const RuntimeTuple runtimeVarParticle(jumpsizeParticle, nbElementsParticle, offsetParticle);
    
    const int jumpsizeFrame = 1;
    const int offsetFrame = 0;
    const int nbElementsFrame = 0;
    
    const RuntimeTuple runtimeSupercell(jumpsizeFrame, nbElementsFrame, offsetFrame);
    /** //////////////////////////////////////////////////////////////////
     * First Test with two loops and unnested Iterator
     *///////////////////////////////////////////////////////////////////
    hzdr::View<Supercell, hzdr::Direction::Forward, hzdr::Collectivity::NonCollectiv, RuntimeTuple> con(&cell, runtimeSupercell);
     
    uint counter(0);
    for(auto it=con.begin(); it!=con.end(); ++it)
    {
        
     //   std::cout << "Hello world" << std::endl;
        auto wrap = *it;
        if(wrap)
        {
            
            auto t = (*wrap);
            hzdr::View<Frame, hzdr::Direction::Forward, hzdr::Collectivity::NonCollectiv,RuntimeTuple> innerCon(&t,runtimeVarParticle);
            for(auto it2=innerCon.begin(); it2 != innerCon.end(); ++it2)
            {
                auto wrapInner = *it2;
                if(wrapInner)
                    counter++;
            }
        }
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42);

    /***************************
     * Second test with a nested Iterator
     * ************************/
    // All Particle within a Supercell
    typedef hzdr::View<Frame, hzdr::Direction::Forward,  hzdr::Collectivity::NonCollectiv,RuntimeTuple> ParticleInFrame;
    
    hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::NonCollectiv, RuntimeTuple,ParticleInFrame> test(cell, runtimeSupercell, ParticleInFrame(nullptr, runtimeVarParticle)); 
    
    counter = 0;
    for(auto it=test.begin(); it!=test.end(); ++it)
    {
        if(*it)
            counter++;
    }
     BOOST_TEST(counter == 42);
    // Second test: 

}

