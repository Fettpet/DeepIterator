#define BOOST_TEST_MODULE UnnestedIterator
#include <boost/test/included/unit_test.hpp>
#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "DeepIterator.hpp"
#include "View.hpp"
#include "Iterator/RuntimeTuple.hpp"
using namespace boost::unit_test;
typedef hzdr::Particle<int_fast32_t, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::SuperCell<Frame> Supercell;
typedef hzdr::SupercellContainer<Supercell> SupercellContainer;
typedef hzdr::runtime::TupleFull RuntimeTuple;

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
    
    typedef hzdr::View<Particle, hzdr::Direction::Forward, hzdr::Collectivity::None, RuntimeTuple> PositionInParticleContainer;
    
    hzdr::View< Frame, 
                    hzdr::Direction::Forward, 
                    
                    hzdr::Collectivity::None, 
                    RuntimeTuple,
                    PositionInParticleContainer > test(cell.firstFrame, 
                                                       runtimeFrame,
                                                       PositionInParticleContainer(nullptr, runtimeVarParticle)); 
                     
    uint counter=0;
    for(auto it=test.begin(); it!=test.end(); ++it)
    {
        if(*it)
        {
            counter += (**it);
            
        }
    }
    
    // sum([0, 19]) = 190
    BOOST_TEST(counter == 190); 
 
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
    hzdr::View<Supercell, hzdr::Direction::Forward, hzdr::Collectivity::None, RuntimeTuple> con(&cell, runtimeSupercell);
     
    auto it=con.begin();
    std:: cout << cell;
    BOOST_TEST((**it).particles[0] == Particle(100, 101));
    uint counter(0);
    for(; it!=con.end(); ++it)
    {
        
     //   std::cout << "Hello world" << std::endl;
        
        auto wrap = *it;
        if(wrap)
        {
            std::cout << **it << std::endl;
            auto t = (*wrap);
            hzdr::View<Frame, hzdr::Direction::Forward, hzdr::Collectivity::None,RuntimeTuple> innerCon(&t,runtimeVarParticle);
            for(auto it2=innerCon.begin(); it2 != innerCon.end(); ++it2)
            {
                std::cout << "T" << std::endl;
                auto wrapInner = *it2;
                if(wrapInner)
                {
                    counter++;
                    std::cout << *wrapInner << std::endl;
                    
                }
            }
        }
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42);

    /***************************
     * Second test with a nested Iterator
     * ************************/
    // All Particle within a Supercell
    typedef hzdr::View<Frame, hzdr::Direction::Forward,  hzdr::Collectivity::None,RuntimeTuple> ParticleInFrame;
    
    hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::None, RuntimeTuple,ParticleInFrame> test(cell, runtimeSupercell, ParticleInFrame(nullptr, runtimeVarParticle)); 
    
    counter = 0;
    for(auto it=test.begin(); it!=test.end(); ++it)
    {
        if(*it)
        {
            counter += (**it).data[0] + (**it).data[1];
            
        }
    }
    // the first position starts at 100
    // sum ( [100, 184] )
     BOOST_TEST(counter == 11886);
    // Second test: 

}

BOOST_AUTO_TEST_CASE(ParticleParticleInteraction)
{
    const int nbSuperCells = 5;
    const int nbFramesInSupercell = 2;
    SupercellContainer supercellContainer(nbSuperCells, nbFramesInSupercell);
    
    typedef hzdr::View<Frame, hzdr::Direction::Forward,  hzdr::Collectivity::None,RuntimeTuple> ParticleInFrame;
    
// create the first iterator
    const int jumpsizeFrame1 = 1;
    const int offsetFrame1 = 0;
    const int nbElementsFrame1 = 0;
    
    const RuntimeTuple runtimeSupercell1(jumpsizeFrame1, nbElementsFrame1, offsetFrame1);
    
    const int jumpsizeParticle1 = 1;
    const int offsetParticle1 = 0;
    const int nbElementsParticle1 = supercellContainer[0].nbParticlesInLastFrame;
    
    const RuntimeTuple runtimeVarParticle1(jumpsizeParticle1, nbElementsParticle1, offsetParticle1);
    
    hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::None, RuntimeTuple,ParticleInFrame> iterSuperCell1(supercellContainer[0], 
                                                                                                                            runtimeSupercell1,
                                                                                                                            ParticleInFrame(nullptr, runtimeVarParticle1)); 
    
// create the second iteartor
    const int jumpsizeFrame2 = 1;
    const int offsetFrame2 = 0;
    const int nbElementsFrame2 = 0;
    
    const RuntimeTuple runtimeSupercell2(jumpsizeFrame2, nbElementsFrame2, offsetFrame2);
    
    const int jumpsizeParticle2 = 1;
    const int offsetParticle2 = 0;
    const int nbElementsParticle2 = supercellContainer[0].nbParticlesInLastFrame;
    
    const RuntimeTuple runtimeVarParticle2(jumpsizeParticle2, nbElementsParticle2, offsetParticle2);
    
    hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::None, RuntimeTuple,ParticleInFrame> iterSuperCell2(supercellContainer[1], 
                                                                                                                            runtimeSupercell2,
                                                                                                                            ParticleInFrame(nullptr, runtimeVarParticle2)); 
    
// first add all 
    for(auto it=iterSuperCell1.begin(); it != iterSuperCell1.end(); ++it)
    {
        if(*it)
        {
            (**it).data[0] = 0;
        }

        for(auto it2 = iterSuperCell2.begin(); it2 != iterSuperCell2.end(); ++it2)
        {
            // check wheter both are valid
            if(*it and *it2)
            {
                (**it).data[0] += (**it2).data[1];
            }
        }
    }
    
// second all particles within the first supercell must have the same value
    for(auto it=iterSuperCell1.begin(); it != iterSuperCell1.end(); ++it)
    {
        for(auto it2 = iterSuperCell1.begin(); it2 != iterSuperCell1.end(); ++it2)
        {
            // check wheter both are valid
            if(*it and *it2)
            {
                BOOST_TEST((**it).data[0] == (**it2).data[0]);
            }
        }
    }
    
}

