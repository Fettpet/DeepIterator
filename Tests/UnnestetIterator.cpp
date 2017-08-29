
#define BOOST_TEST_MODULE UnnestedIterator
#include <boost/test/included/unit_test.hpp>

#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/SupercellContainer.hpp"
#include "PIC/Particle.hpp"
#include "DeepIterator.hpp"
#include "View.hpp"
#include "Definitions/hdinline.hpp"
using namespace boost::unit_test;
typedef hzdr::Particle<int_fast32_t, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::SuperCell<Frame> Supercell;

/**
 * @brief This test iterate over all Positions in a Particle
 */

BOOST_AUTO_TEST_CASE(PositionsInFrameNotCollectiv)
{

    
    Particle particle1(1,4);
/**
 *@brief 1. test: Forward with jumpsize 1
 */ 

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    // over each element                          
    auto it = makeIterator(particle1, 
                               makeAccessor(particle1), 
                               makeNavigator(particle1, 
                                              hzdr::Direction::Forward(),
                                              Offset(0),
                                              Jumpsize(1)));
    std::cout << "Particle" << particle1 << std::endl;
  //  hzdr::View<Particle, hzdr::Direction::Forward<1>> con(&test1);
    auto itTest = it+1;
    
   
     BOOST_TEST(*it == 1);
    ++it;
    // check wheter ++ and == work
    BOOST_TEST((it == itTest));
    
    
    
    BOOST_TEST(*it == 4);
    
    ++it;
    BOOST_TEST(it.isAtEnd());
    

/**
 *@brief 2. test: Backward 
*/   

    auto it2 = makeIterator(particle1, 
                               makeAccessor(particle1), 
                               makeNavigator(particle1, 
                                              hzdr::Direction::Backward(),
                                              Offset(0),
                                              Jumpsize(1)));

    BOOST_TEST(*it2 == 4);
    ++it2;
    BOOST_TEST(*it2 == 1);
    ++it2;
    BOOST_TEST(it2.isAtEnd());

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
        typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    /** ******************
    // @brief 1. Test Forward with Jumpsize 1 nonCollectiv
    ********************/

    // the 2 is the number of elements in Last Frame
    auto && it = makeIterator(*(cell.firstFrame), 
                               makeAccessor(*(cell.firstFrame)),
                               makeNavigator(*(cell.firstFrame),
                                              hzdr::Direction::Forward(),
                                              Offset(0),
                                              Jumpsize(1)));

    int_fast32_t counter(0);
    for(; not it.isAtEnd(); ++it)
    {   
            counter += (*it).data[0] + (*it).data[1];
        
    }
    // sum [0, 19]
    BOOST_TEST(counter == 190);

/** ******************
@brief 2. Test Forward with Jumpsize 3 nonCollectiv
We implement a own collectivity class
********************/

        auto && it2 = makeIterator(*(cell.firstFrame), 
                               makeAccessor(*(cell.firstFrame)),
                               makeNavigator(*(cell.firstFrame),
                                              hzdr::Direction::Forward(),
                                              Offset(0),
                                              Jumpsize(3)));

    counter = 0;
    for(; not it2.isAtEnd(); ++it2)
    {   
            counter += (*it2).data[0] + (*it2).data[1];
        
    }
//     // the first particle (0, 1), fourth ( 6, 7) seventh ( 12, 13) and last (18, 19) add together
    BOOST_TEST(counter == 76);
//   
//     /** ******************
//     // @brief 3. Test backward with Jumpsize 3 nonCollectiv
//     ********************/
// 
        auto && it3 = makeIterator(*(cell.firstFrame), 
                               makeAccessor(*(cell.firstFrame)),
                               makeNavigator(*(cell.firstFrame),
                                              hzdr::Direction::Backward(),
                                              Offset(0),
                                              Jumpsize(3)));
    counter = 0;

    for(; not it3.isAtEnd(); ++it3)
    {   
            counter += (*it3).data[0] + (*it3).data[1];
        
    }
    // the first particle (0, 1), fourth ( 6, 7) seventh ( 12, 13) and last (18, 19) add together
    BOOST_TEST(counter == 76);


}

