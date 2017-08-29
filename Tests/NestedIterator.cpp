

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
    
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    auto && it = hzdr::makeIterator(*(cell.firstFrame), 
                               hzdr::makeAccessor(*(cell.firstFrame)),
                               hzdr::makeNavigator(*(cell.firstFrame), 
                                              hzdr::Direction::Forward(),
                                              Offset(0),
                                              Jumpsize(1)),
                               hzdr::make_child(hzdr::makeAccessor(),
                                                hzdr::makeNavigator(hzdr::Direction::Forward(),
                                                Offset(0),
                                                Jumpsize(1))));

    uint counter=0;
    for(; not it.isAtEnd(); ++it)
    {
        counter += (*it);
    }
//     
//     // sum([0, 19]) = 190
    BOOST_TEST((counter == 190)); 
// //  
}


BOOST_AUTO_TEST_CASE(ParticleInSuperCell)
{


    Supercell cell(5, 2);
    /** //////////////////////////////////////////////////////////////////
     * First Test with two loops and unnested Iterator
     *///////////////////////////////////////////////////////////////////
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    auto && it = hzdr::makeIterator(cell, 
                               hzdr::makeAccessor(cell),
                               hzdr::makeNavigator(cell,
                                              hzdr::Direction::Forward(),
                                              Offset(0),
                                              Jumpsize(1)),
                               hzdr::make_child(hzdr::makeAccessor(),
                                                hzdr::makeNavigator(hzdr::Direction::Forward(),
                                                Offset(0),
                                                Jumpsize(1))));
     
    

    BOOST_TEST((*it) == Particle(100, 101));
    uint counter(0);
    for(; not it.isAtEnd(); ++it)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42);

}


BOOST_AUTO_TEST_CASE(PositionsInSupercell)
{


    Supercell cell(5, 2);
    /** //////////////////////////////////////////////////////////////////
     * First Test with two loops and unnested Iterator
     *///////////////////////////////////////////////////////////////////
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    auto && it = hzdr::makeIterator(cell, 
                               hzdr::makeAccessor(cell),
                               hzdr::makeNavigator(cell,
                                              hzdr::Direction::Forward(),
                                              Offset(0),
                                              Jumpsize(1)),
                                    hzdr::make_child(
                                                hzdr::makeAccessor(),
                                                hzdr::makeNavigator(hzdr::Direction::Forward(),
                                                    Offset(0),
                                                    Jumpsize(1)),
                                                hzdr::make_child(
                                                    hzdr::makeAccessor(),
                                                    hzdr::makeNavigator(hzdr::Direction::Forward(),
                                                        Offset(0),
                                                        Jumpsize(1)))));
     
    

    uint counter(0);
    for(; not it.isAtEnd(); ++it)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84);

}


BOOST_AUTO_TEST_CASE(ParticleParticleInteraction)
{
//     const int nbSuperCells = 5;
//     const int nbFramesInSupercell = 2;
//     SupercellContainer supercellContainer(nbSuperCells, nbFramesInSupercell);
//     
//     typedef hzdr::View<Frame, hzdr::Direction::Forward<1> > ParticleInFrame;
//     
// 
//     
//     hzdr::View<Supercell, hzdr::Direction::Forward<1>, ParticleInFrame> iterSuperCell1(supercellContainer[0]); 
// // create the second iteartor
// 
//     
//     hzdr::View<Supercell, hzdr::Direction::Forward<1>, ParticleInFrame> iterSuperCell2(supercellContainer[1]);
// // first add all 
//     for(auto it=iterSuperCell1.begin(); it != iterSuperCell1.end(); ++it)
//     {
//         if(*it)
//         {
//             (**it).data[0] = 0;
//         }
// 
//         for(auto it2 = iterSuperCell2.begin(); it2 != iterSuperCell2.end(); ++it2)
//         {
//             // check wheter both are valid
//             if(*it and *it2)
//             {
//                 (**it).data[0] += (**it2).data[1];
//             }
//         }
//     }
//     
// // second all particles within the first supercell must have the same value
//     for(auto it=iterSuperCell1.begin(); it != iterSuperCell1.end(); ++it)
//     {
//         for(auto it2 = iterSuperCell1.begin(); it2 != iterSuperCell1.end(); ++it2)
//         {
//             // check wheter both are valid
//             if(*it and *it2)
//             {
//                 BOOST_TEST((**it).data[0] == (**it2).data[0]);
//             }
//         }
//     }
    
}


BOOST_AUTO_TEST_CASE(ParticlesWithSimulatedThreads)
{
    /**
     * First Test, check whether the right number of invalid objects are detected
     * We have a supercell with 5 frames. The first four frames have 10 particles.
     * The last frame has two particles. I.e there are 42 particles. We iterate 
     * over all particles.
     */
    Supercell cell(5, 2);
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    for(int nbThreads = 2; nbThreads <=5; ++nbThreads)
    {

        int count = 0;
        auto myId = 1;

        auto && it = hzdr::makeIterator(
                        cell,
                        hzdr::makeAccessor(cell),
                        hzdr::makeNavigator(
                            cell, 
                            hzdr::Direction::Forward(),
                            Offset(0),
                            Jumpsize(1)),
                        hzdr::make_child(
                            hzdr::makeAccessor(),
                            hzdr::makeNavigator(
                                hzdr::Direction::Forward(),
                                Offset(myId),
                                Jumpsize(nbThreads))));
            
       
        for(; not it.isAtEnd(); ++it)
        {
            ++count;
        }   
        if(nbThreads == 2)
        {
            // 1,3,5,7,9
            BOOST_TEST(count == 5 * 4 + 1);
        }
        if(nbThreads == 3)
            // 1,4,7
            BOOST_TEST(count == 3*4 + 1);
        if(nbThreads == 4)
            // 1, 5, 9
            BOOST_TEST(count == 3 *4 + 1);
        if(nbThreads == 5)
            // 1, 6
            BOOST_TEST(count == 2 * 4 + 1);
        
        /**
         * second test, write and check the result
         * 
         */
        for(auto myId=0; myId<nbThreads;++myId)
        {
            auto it2 = hzdr::makeIterator(
                                cell,
                                hzdr::makeAccessor(cell),
                                hzdr::makeNavigator(
                                    cell, 
                                    hzdr::Direction::Forward(),
                                    Offset(0),
                                    Jumpsize(1)),
                                hzdr::make_child(
                                    hzdr::makeAccessor(),
                                    hzdr::makeNavigator(
                                        hzdr::Direction::Forward(),
                                        Offset(myId),
                                        Jumpsize(nbThreads))));
            for(; not it2.isAtEnd(); ++it2)
            {
                //syncThreads() // geht nicht wegen deadlock
                auto value = (*it2).data[1] == -1;
                if((*it2).data[1] == -1)
                {
                    it2.childIterator.isAtEnd();
                }
                (*it2).data[0] = myId;
                BOOST_TEST((*it2).data[0] == myId);
                
            }       
        }
    }

}


