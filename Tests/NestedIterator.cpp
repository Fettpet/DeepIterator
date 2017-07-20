
#define BOOST_TEST_MODULE NestedIterator
#include <boost/test/included/unit_test.hpp>

#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "DeepIterator.hpp"
#include "View.hpp"
#include "Definitions/hdinline.hpp"

//using namespace boost::unit_test;

typedef hzdr::Particle<int_fast32_t, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::SuperCell<Frame> Supercell;
typedef hzdr::SupercellContainer<Supercell> SupercellContainer;


BOOST_AUTO_TEST_CASE(PositionsInFrames)
{
 
    Supercell cell(5, 2);
    

    

    typedef hzdr::View<Particle, hzdr::Direction::Forward<1> > PositionInParticleContainer;
    
    hzdr::View< Frame, 
                    hzdr::Direction::Forward<1>, 
                    PositionInParticleContainer > test(cell.firstFrame); 
                     
    uint counter=0;
    for(auto it=test.begin(); it!=test.end(); ++it)
    {
        if(*it)
        {
            counter += (**it);
            
        }
    }
//     
//     // sum([0, 19]) = 190
    BOOST_TEST((counter == 190)); 
//  
}


BOOST_AUTO_TEST_CASE(ParticleInSuperCell)
{


    Supercell cell(5, 2);
    /** //////////////////////////////////////////////////////////////////
     * First Test with two loops and unnested Iterator
     *///////////////////////////////////////////////////////////////////
    hzdr::View<Supercell, hzdr::Direction::Forward<1> > con(&cell);
     
    auto it=con.begin();

    BOOST_TEST((**it).particles[0] == Particle(100, 101));
    uint counter(0);
    for(; it!=con.end(); ++it)
    {
        
     //   std::cout << "Hello world" << std::endl;
        
        auto wrap = *it;
        if(wrap)
        {

            auto t = (*wrap);
            hzdr::View<Frame, hzdr::Direction::Forward<1> > innerCon(&t);
            for(auto it2=innerCon.begin(); it2 != innerCon.end(); ++it2)
            {
                auto wrapInner = *it2;
                if(wrapInner)
                {
                    counter++;
                    
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
    typedef hzdr::View<Frame, hzdr::Direction::Forward<1> > ParticleInFrame;
    
    hzdr::View<Supercell, hzdr::Direction::Forward<1>, ParticleInFrame> test(cell); 
    
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
    
    typedef hzdr::View<Frame, hzdr::Direction::Forward<1> > ParticleInFrame;
    

    
    hzdr::View<Supercell, hzdr::Direction::Forward<1>, ParticleInFrame> iterSuperCell1(supercellContainer[0]); 
// create the second iteartor

    
    hzdr::View<Supercell, hzdr::Direction::Forward<1>, ParticleInFrame> iterSuperCell2(supercellContainer[1]);
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

// #if 0
// BOOST_AUTO_TEST_CASE(ParticlesWithSimulatedThreads)
// {
//     /**
//      * First Test, check whether the right number of invalid objects are detected
//      * 
//      */
//     Supercell cell(5, 2);
// 
//     for(int nbThreads = 2; nbThreads <=5; ++nbThreads)
//     {
// 
//         int count = 0;
//         for(int i=0; i<nbThreads; ++i)
//         {   
//             const int jumpsizeFrame2 = 1;
//             const int offsetFrame2 = 0;
//             const int nbElementsFrame2 = 0;
//             
//             const RuntimeTuple runtimeSupercell2(offsetFrame2, nbElementsFrame2, jumpsizeFrame2);
//             
//             const int jumpsizeParticle2 = nbThreads;
//             const int offsetParticle2 = i;
//             const int nbElementsParticle2 = cell.nbParticlesInLastFrame;
//             
//             const RuntimeTuple runtimeVarParticle2(offsetParticle2, nbElementsParticle2, jumpsizeParticle2);
//             
//             hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::None, RuntimeTuple,ParticleInFrame> iterSuperCell(cell, 
//                                                                                                                                     runtimeSupercell2,
//                                                                                                                                     ParticleInFrame(nullptr, runtimeVarParticle2)); 
//             for(auto it=iterSuperCell.begin(); it != iterSuperCell.end(); ++it)
//             {
//                 if(not *it)
//                 {
//                     ++count;
//                 }
//             }   
//         }
//         if(nbThreads == 2)
//             BOOST_TEST(count == 4);
//         
//         if(nbThreads == 3)
//             BOOST_TEST(count == 5);
//         if(nbThreads == 4)
//             BOOST_TEST(count == 12);
//         if(nbThreads == 5)
//             BOOST_TEST(count == 3);
//     }
// 
// /**
//  * Second test: 
//  */
// }
// #endif
// 
// 
