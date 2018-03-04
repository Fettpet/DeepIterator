

#define BOOST_TEST_MODULE UnnestedIterator
#include <boost/test/included/unit_test.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"

using namespace boost::unit_test;
typedef int_fast32_t ParticleProperty; 
typedef hzdr::Particle<ParticleProperty, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::Supercell<Frame> Supercell;

/**
 * @brief This test iterate over all Positions in a Particle
 */

BOOST_AUTO_TEST_CASE(PositionsInParticlesNotCollectiv)
{
    
    // We particle with two properties
    ParticleProperty property1 = static_cast<ParticleProperty>(1);
    ParticleProperty property2 = static_cast<ParticleProperty>(4);
    Particle particle1(property1, property2);
/**
 *@brief 1. test: Forward with jumpsize 1
 */ 

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    // over each element                          
    auto view = hzdr::makeView(particle1, 
                               hzdr::makeIteratorPrescription(   
                                    hzdr::makeAccessor(), 
                                    hzdr::makeNavigator(
                                        Offset(0u),
                                        Jumpsize(1u))));
  //  hzdr::View<Particle, hzdr::Direction::Forward<1>> con(&test1);
    auto it = view.begin();
    
   
    BOOST_TEST(*it == property1);
    ++it;
    // check wheter ++ and == work
    
    
    
    BOOST_TEST(*it == property2);
    
    // Check end
    ++it;
    BOOST_TEST((view.end() == it));
    

/**
 *@brief 2. test: Backward 
*/   

    auto it2 = view.rbegin();

    BOOST_TEST(*it2 == property2);
    --it2;
    BOOST_TEST(*it2 == property1);
    --it2;
    BOOST_TEST((view.rend() == it2));

    it2 += 2;
    --it;
    BOOST_TEST((it2 == it));
}

/**
 * @brief This test iterate over all particles within a frame. We test non 
 * collectiv functionality. There are several tests
 * 1. Forward with jumpsize 1
 * 2. Forward with jumpsize 3
 */
BOOST_AUTO_TEST_CASE(ParticleInFrameNotCollectiv)
{

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);
    /** ******************
    // @brief 1. Test Forward with Jumpsize 1 nonCollectiv
    ********************/

    // the 2 is the number of elements in Last Frame
    auto && view = hzdr::makeView(*(supercell.firstFrame), 
                                  hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0u),
                                            Jumpsize(1u))));

    int_fast32_t counter(0);
    for(auto && it = view.begin(); it != view.end(); ++it)
    {   
            counter += (*it).data[0u] + (*it).data[1u];
        
    }
    // sum [0, 19]
    BOOST_TEST(counter == 190);

/** ******************
@brief 2. Test Forward with Jumpsize 3 nonCollectiv
We implement a own collectivity class
********************/


    counter = 0;
    for(auto && it = view.begin(); it != view.end(); it+=3)
    {   
            counter += (*it).data[0u] + (*it).data[1u];
        
    }
    
    auto && viewJump3 = hzdr::makeView(*(supercell.firstFrame), 
                                hzdr::makeIteratorPrescription(
                                    hzdr::makeAccessor(),
                                    hzdr::makeNavigator(
                                        Offset(0u),
                                        Jumpsize(3u))));
    auto counterJump3 = 0;
    for(auto && it = viewJump3.begin(); it != viewJump3.end(); ++it)
    {   
            counterJump3 += (*it).data[0u] + (*it).data[1u];
        
    }
    BOOST_TEST((counter == counterJump3));
  
    /** ******************
    @brief 3. Test backward with Jumpsize 3 nonCollectiv. We can not test 
    whether counterBackwardJump3 == counter, since they iterate over different 
    positions. See this little example with 10 particles within a frame:
    [0,1,2,3,4,6,7,8,9]. The first case is the iteration from begin to the end 
    [x, , ,x, , ,x, , ]. The x stands for access to these value. The second case
    is the iteration from rbegin to rend: [ , ,x, , ,x, , ,x].
    ********************/
    std::cout << *(supercell.firstFrame) << std::endl;
    std::cout << "Values from it" << std::endl;
    counter = 0;
    for(auto && it=view.rbegin(); it != view.rend(); it -= 3)
    {   
        std::cout << "\t" << *it << std::endl;
            counter += (*it).data[0u] + (*it).data[1u];
    }
    std::cout << "Values from it3" << std::endl;
    auto counterBackwardJump3 = 0;
    for(auto && it=viewJump3.rbegin(); it != viewJump3.rend(); --it)
    {   
        std::cout << "\t" << *it << std::endl;
            counterBackwardJump3 += (*it).data[0u] + (*it).data[1u];
    }
    // the first particle (0, 1), fourth ( 6, 7) seventh ( 12, 13) and last (18, 19) add together
    BOOST_TEST(counter == counterBackwardJump3);
}

/**
 * These example is importent to check the doubly link list like iterator. The 
 * iterator is a bidirectiional one.
 */
BOOST_AUTO_TEST_CASE(FramesInSupercells)
{
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    // Supercell with 5 frames. In the last frame are 2 particles
    uint_fast32_t const nbFrames = 5u;
    uint_fast32_t const nbParticlesInLastFrame = 2u;
    Supercell supercell(nbFrames, nbParticlesInLastFrame);

    
    
    auto && view = hzdr::makeView(
        supercell,
        hzdr::makeIteratorPrescription(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                Offset(0u),
                Jumpsize(1u))));
    
    // we count the number of frames
    auto counter = 0u;
    for(auto it=view.begin(); it!=view.end(); ++it)
    {
        ++counter;
    }
    BOOST_TEST(counter == nbFrames);
    
    // we count the number of frames reverse
    auto counterReverse = 0u;
    for(auto it=view.rbegin(); it!=view.rend(); --it)
    {
        ++counterReverse;
    }
    BOOST_TEST(counterReverse == nbFrames);
    
    // we count each second frame
    
    auto && viewJump2 = hzdr::makeView(
        supercell,
        hzdr::makeIteratorPrescription(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                Offset(0u),
                Jumpsize(2u))));
    
    auto counterJump2 = 0u;
    for(auto it=viewJump2.begin(); it!=viewJump2.end(); ++it)
    {
        counterJump2++;
    }
    BOOST_TEST((counterJump2 == ((nbFrames + 1u)/2u) ));
    
    auto counterJump2Reverse = 0u;
    for(auto it=viewJump2.rbegin(); it!=viewJump2.rend(); --it)
    {
        counterJump2Reverse++;
    }
    BOOST_TEST((counterJump2Reverse == ((nbFrames + 1u)/2u) ));
}   


