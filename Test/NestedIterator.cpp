
#define BOOST_TEST_MODULE NestedIterator
#include <boost/test/included/unit_test.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"

//using namespace boost::unit_test;
typedef int_fast32_t ParticleProperty;
typedef hzdr::Particle<ParticleProperty, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::Supercell<Frame> Supercell;
typedef hzdr::SupercellContainer<Supercell> SupercellContainer;


BOOST_AUTO_TEST_CASE(PositionsInFrames)
{
 
    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);
    
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
/**
 * @brief The first test is used to verify the iteration over all particle 
 * attributes within a frame. 
 */
    auto && view = hzdr::makeView(
                        *(supercell.firstFrame), 
                        hzdr::makeIteratorPrescription(
                            hzdr::makeAccessor(),
                            hzdr::makeNavigator( 
                                Offset(0),
                                Jumpsize(1)),
                            hzdr::makeIteratorPrescription(
                                hzdr::makeAccessor(),
                                hzdr::makeNavigator(
                                    Offset(0),
                                    Jumpsize(1)))));
                            
    std::cout << *(supercell.firstFrame) << std::endl;
    // 1. test ++it
    uint counter = 0u;
    for(auto && it = view.begin(); it != view.end(); ++it)
    {
        counter += (*it);
    }
    // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u); 
    
    // 2. test it++
    counter = 0;
    for(auto && it = view.begin(); it != view.end(); it++)
    {
        counter += (*it);
    }
    // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u); 
    
    // 3. test --it
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); --it)
    {
        std::cout << *it << std::endl;
        counter += (*it);
    }
    // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u); 
    
    // 4. test it--
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it--)
    {
        counter += (*it);
    }

//     // sum([0, 19]) = 190
    BOOST_TEST(counter == 190u); 
}


BOOST_AUTO_TEST_CASE(ParticleInSupercell)
{


    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);
    
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    auto && view = hzdr::makeView( 
                               supercell,
                               hzdr::makeIteratorPrescription(
                                    hzdr::makeAccessor(),
                                    hzdr::makeNavigator(
                                        Offset(0),
                                        Jumpsize(1)),
                                    hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)))));
     
    
   // 1. test ++it
    uint counter = 0u;
    for(auto && it = view.begin(); it != view.end(); ++it)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);
    
    // 2. test it++
    counter = 0u;
    for(auto && it = view.begin(); it != view.end(); it++)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);
    
    // 3. test --it
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); --it)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);
    
    // 4. test it--
    counter = 0u;
    for(auto && it = view.rbegin(); it != view.rend(); it--)
    {
        counter++;
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 42u);


}


BOOST_AUTO_TEST_CASE(PositionsInSupercell)
{
/**
 * @brief We use this test for 3 layer. 
 */

    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    
    auto && view = hzdr::makeView(
        supercell, 
        makeIteratorPrescription(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                Offset(0),
                Jumpsize(1)),
            hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(0),
                    Jumpsize(1)),
                hzdr::makeIteratorPrescription(
                    hzdr::makeAccessor(),
                    hzdr::makeNavigator(
                        Offset(0),
                        Jumpsize(1))))));
     
    std::cout << "Data structure: " << std::endl << supercell << std::endl;
    // test ++it
    uint counter = 0u;
    for(auto it = view.begin(); it != view.end(); ++it)
    {
        std::cout << *it << std::endl;
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);
    
    // test it++
    counter = 0u;
    for(auto it = view.begin(); it != view.end(); it++)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);
    
    //test --it
    counter = 0u;
    for(auto it = view.rbegin(); it != view.rend(); --it)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);

    //test it--
    counter = 0u;
    for(auto it = view.rbegin(); it != view.rend(); it--)
    {
        counter++;            
    }
    // There are 4 full frames with 10 Elements an one frame with 2 elements
    BOOST_TEST(counter == 84u);
}

#if 0
BOOST_AUTO_TEST_CASE(ParticleParticleInteraction)
{
    const int nbSupercells = 5;
    const int nbFramesInSupercell = 2;
    SupercellContainer supercellContainer(nbSupercells, nbFramesInSupercell);
    
    typedef hzdr::View<Frame, hzdr::Direction::Forward<1> > ParticleInFrame;
    

    
    hzdr::View<Supercell, hzdr::Direction::Forward<1>, ParticleInFrame> iterSupercell1(supercellContainer[0]); 
// create the second iteartor

    
    hzdr::View<Supercell, hzdr::Direction::Forward<1>, ParticleInFrame> iterSupercell2(supercellContainer[1]);
// first add all 
    for(auto it=iterSupercell1.begin(); it != iterSupercell1.end(); ++it)
    {
        if(*it)
        {
            (**it).data[0] = 0;
        }

        for(auto it2 = iterSupercell2.begin(); it2 != iterSupercell2.end(); ++it2)
        {
            // check wheter both are valid
            if(*it and *it2)
            {
                (**it).data[0] += (**it2).data[1];
            }
        }
    }
    
// second all particles within the first supercell must have the same value
    for(auto it=iterSupercell1.begin(); it != iterSupercell1.end(); ++it)
    {
        for(auto it2 = iterSupercell1.begin(); it2 != iterSupercell1.end(); ++it2)
        {
            // check wheter both are valid
            if(*it and *it2)
            {
                BOOST_TEST((**it).data[0] == (**it2).data[0]);
            }
        }
    }
    
}*/


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
#endif


