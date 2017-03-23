#if 0
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
/**
 * @brief This test iterate over all Positions in a Particle
 */
BOOST_AUTO_TEST_CASE(PositionsInFrameNonCollectiv)
{

    
    Particle test1(1,4);
/**
 *@brief 1. test: Forward with jumpsize 1
 */ 


    // over each element                          
    const int jumpsize = 1;
    const int offset = 0;
    const int nbElements = 2;
    
    const RuntimeTuple runtimeVar(jumpsize, nbElements, offset);
    hzdr::View<Particle, hzdr::Direction::Forward, hzdr::Collectivity::NonCollectiv, RuntimeTuple> con(&test1, runtimeVar);
    auto it = con.begin();
    auto wrap = *it;
    BOOST_TEST(*wrap == 1);
    ++it;
    wrap = *it;
    BOOST_TEST(*wrap == 4);
    
    ++it;
    BOOST_TEST(not(it != con.end()));
    

/**
 *@brief 2. test: Backward with jumpsize 2

    hzdr::View<Particle, hzdr::Direction::Backward, 2, hzdr::Collectivity::NonCollectiv, RuntimeTuple> con2(&test1, static_cast<uint_fast32_t>(2));
    auto it2 = con2.begin();
    BOOST_TEST(*(*it2) == 4);
    ++it2;
    // since there are only two elements, its at the end
    BOOST_TEST(not (it2 != con2.end()));
   */   
}


/**
 * @brief This test iterate over all particles within a frame. We test non 
 * collectiv functionality. There are several tests
 * 1. Forward with jumpsize 1
 * 2. Forward with jumpsize 3
 * 3. Backward with jumpsize 6
 */
BOOST_AUTO_TEST_CASE(ParticleInFrameNonCollectiv)
{

    Supercell cell(5, 2);
    
    /** ******************
    // @brief 1. Test Forward with Jumpsize 1 nonCollectiv
    ********************/
       // over each element                          
    const int jumpsize = 1;
    const int offset = 0;
    const int nbElements = 2;
    
    const RuntimeTuple runtimeVar(jumpsize, nbElements, offset); 
    // the 2 is the number of elements in Last Frame
    hzdr::View<Frame, hzdr::Direction::Forward, hzdr::Collectivity::NonCollectiv, RuntimeTuple> con(*cell.firstFrame, runtimeVar);

    uint_fast32_t counter(0);
    for(auto it = con.begin(); it != con.end(); ++it)
    {   
        auto wrap = *it;
        if(wrap)
        {
            counter++;
        }
    }
    BOOST_TEST(counter == 10);
    /** ******************
    // @brief 2. Test Forward with Jumpsize 3 nonCollectiv
    ********************/
           // over each element                          
    const int jumpsize2 = 3;
    const int offset2 = 0;
    const int nbElements2 = 2;
    
    const RuntimeTuple runtimeVar2(jumpsize2, nbElements2, offset2); 
    hzdr::View<Frame, hzdr::Direction::Forward, hzdr::Collectivity::NonCollectiv, RuntimeTuple> con2(*cell.firstFrame, runtimeVar2);

    counter = 0;
    for(auto it = con2.begin(); it != con2.end(); ++it)
    {   
        auto wrap = *it;
        if(wrap)
        {
            counter++;
        }
    }
    BOOST_TEST(counter == 4);
    
    /** ******************
    // @brief 3. Test backward with Jumpsize 3 nonCollectiv
    ********************/
               // over each element                          
    const int jumpsize3 = 3;
    const int offset3 = 0;
    const int nbElements3 = 10;
    
    const RuntimeTuple runtimeVar3(jumpsize3, nbElements3, offset3); 
    hzdr::View<Frame, hzdr::Direction::Backward, hzdr::Collectivity::NonCollectiv, RuntimeTuple> con3(*cell.firstFrame, runtimeVar3);
    counter = 0;
    for(auto it = con3.begin(); it != con3.end(); ++it)
    {   
        auto wrap = *it;
        if(wrap)
        {
            counter++;
        }
    }
    BOOST_TEST(counter == 4);


}

BOOST_AUTO_TEST_CASE(FrameInSuperCell)
{
    typedef hzdr::Particle<int, 2> Particle;
    typedef hzdr::Frame<Particle, 10> Frame;
    typedef hzdr::SuperCell<Frame> Supercell;
    Supercell cell(5, 2);
    
    const int jumpsize3 = 1;
    const int offset3 = 0;
    const int nbElements3 = 2;
    
    const RuntimeTuple runtimeVar2(jumpsize3, nbElements3, offset3); 
    hzdr::View<Supercell, hzdr::Direction::Forward, hzdr::Collectivity::NonCollectiv, RuntimeTuple> con(&cell, runtimeVar2);

    uint_fast32_t counter(0);
    for(auto it=con.begin(); it!=con.end(); ++it)
    {
        auto wrap = *it;
        if(wrap)
        {
            ++counter;
        }
    }
    BOOST_TEST(counter == 5);
}
#endif