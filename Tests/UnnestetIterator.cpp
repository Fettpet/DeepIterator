
#define BOOST_TEST_MODULE UnnestedIterator
#include <boost/test/included/unit_test.hpp>

#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/SupercellContainer.hpp"
#include "PIC/Particle.hpp"
#include "DeepIterator.hpp"
#include "View.hpp"
#include "Iterator/RuntimeTuple.hpp"
#include "Definitions/hdinline.hpp"
using namespace boost::unit_test;
typedef hzdr::Particle<int_fast32_t, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::SuperCell<Frame> Supercell;

typedef hzdr::runtime::TupleFull RuntimeTuple;
/**
 * @brief This test iterate over all Positions in a Particle
 */

BOOST_AUTO_TEST_CASE(PositionsInFrameNotCollectiv)
{

    
    Particle test1(1,4);
/**
 *@brief 1. test: Forward with jumpsize 1
 */ 


    // over each element                          
    const int jumpsize = 1;
    const int offset = 0;
    const int nbElements = 2;
    
    const RuntimeTuple runtimeVar(offset, nbElements, jumpsize);
    hzdr::View<Particle, hzdr::Direction::Forward, hzdr::Collectivity::None, RuntimeTuple> con(&test1, runtimeVar);
    auto it = con.begin();
    auto itTest = it+1;
    auto wrap = *it;
    
   
    BOOST_TEST(*wrap == 1);
    ++it;
    // check wheter ++ and == work
    BOOST_TEST((it == itTest));
    
    
    wrap = *it;
    BOOST_TEST(*wrap == 4);
    
    ++it;
    auto wrapper = *it;
    bool result(wrapper);
    BOOST_TEST(not result);
    ++it;
    BOOST_TEST(not (it != con.end()));
    

/**
 *@brief 2. test: Backward 
   */   
    const int jumpsize2 = 2;
    const int offset2 = 0;
    const int nbElements2 = 2;
    const RuntimeTuple runtimeVar2(offset2, nbElements2, jumpsize2);
    hzdr::View<Particle, hzdr::Direction::Backward,  hzdr::Collectivity::None, RuntimeTuple> con2(&test1, runtimeVar2);
    auto it2 = con2.begin();
    BOOST_TEST(*(*it2) == 4);

}

/**
 * @brief This test iterate over all particles within a frame. We test non 
 * collectiv functionality. There are several tests
 * 1. Forward with jumpsize 1
 * 2. Forward with jumpsize 3
 * 3. Backward with jumpsize 6
 */
BOOST_AUTO_TEST_CASE(ParticleInFrameNotCollectiv)
{

    Supercell cell(5, 2);
    
    /** ******************
    // @brief 1. Test Forward with Jumpsize 1 nonCollectiv
    ********************/
       // over each element                          
    const int jumpsize = 1;
    const int offset = 0;
    const int nbElements = 2;
    
    const RuntimeTuple runtimeVar(offset, nbElements, jumpsize); 
    // the 2 is the number of elements in Last Frame
    hzdr::View<Frame, hzdr::Direction::Forward, hzdr::Collectivity::None, RuntimeTuple> con(*cell.firstFrame, runtimeVar);

    int_fast32_t counter(0);
    for(auto it = con.begin(); it != con.end(); ++it)
    {   
        auto wrap = *it;
        if(wrap)
        {

            counter += (*wrap).data[0] + (*wrap).data[1];
        }
    }
    // sum [0, 19]
    BOOST_TEST(counter == 190);

    /** ******************
    // @brief 2. Test Forward with Jumpsize 3 nonCollectiv
    ********************/
           // over each element                          
    const int jumpsize2 = 3;
    const int offset2 = 0;
    const int nbElements2 = 2;
    
    const RuntimeTuple runtimeVar2(offset2, nbElements2, jumpsize2); 
    hzdr::View<Frame, hzdr::Direction::Forward, hzdr::Collectivity::None, RuntimeTuple> con2(*cell.firstFrame, runtimeVar2);

    counter = 0;
    for(auto it = con2.begin(); it != con2.end(); ++it)
    {   
        auto wrap = *it;
            
        if(wrap)
        {

            counter += (*wrap).data[0] + (*wrap).data[1];
        }
    }
    // the first particle (0, 1), fourth ( 6, 7) seventh ( 12, 13) and last (18, 19) add together
    BOOST_TEST(counter == 76);
  
    /** ******************
    // @brief 3. Test backward with Jumpsize 3 nonCollectiv
    ********************/
               // over each element                          
    const int jumpsize3 = 3;
    const int offset3 = 0;
    const int nbElements3 = 10;
    
    const RuntimeTuple runtimeVar3(offset3, nbElements3, jumpsize3); 
    hzdr::View<Frame, hzdr::Direction::Backward, hzdr::Collectivity::None, RuntimeTuple> con3(*cell.firstFrame, runtimeVar3);
    counter = 0;
    for(auto it = con3.begin(); it != con3.end(); ++it)
    {   
        auto wrap = *it;
        if(wrap)
        {
            counter += (*wrap).data[0] + (*wrap).data[1];
        }
    }
    // the first particle (0, 1), fourth ( 6, 7) seventh ( 12, 13) and last (18, 19) add together
    BOOST_TEST(counter == 76);


}

BOOST_AUTO_TEST_CASE(FrameInSuperCellNotCollectiv)
{

    Supercell cell(5, 2);
    
    const int jumpsize3 = 1;
    const int offset3 = 0;
    const int nbElements3 = 2;
    
    const RuntimeTuple runtimeVar2(offset3, nbElements3 ,jumpsize3); 
    hzdr::View<Supercell, hzdr::Direction::Forward, hzdr::Collectivity::None, RuntimeTuple> con(&cell, runtimeVar2);

    int_fast32_t counter(0);
    for(auto it=con.begin(); it!=con.end(); ++it)
    {
        ++counter;
        auto wrap = *it;
        if(wrap)
        {
            
        }
    }
    BOOST_TEST(counter == 5);
}
