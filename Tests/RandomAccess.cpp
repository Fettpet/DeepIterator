
/**
 * @author Sebastian Hahn t.hahn <at> hzdr.de
 * @brief Within these test collection we need to test the following operations:
 * 1. it+=n // done
 * 2. it-=n // done
 * 5. it1 > it2
 * 6. it1 < it2
 * 7. it1 <= it2
 * 8. it2 >= it2
 * 9. it[n]
 * 
 */

#define BOOST_TEST_MODULE RandomAccessIterator
#include <boost/test/included/unit_test.hpp>

#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "View.hpp"
#include "DeepIterator.hpp"
#include "View.hpp"
#include "Definitions/hdinline.hpp"
typedef hzdr::Particle<int_fast32_t, 2u> Particle;
typedef hzdr::Frame<Particle, 10u> Frame;
typedef hzdr::Supercell<Frame> Supercell;
typedef hzdr::SupercellContainer<Supercell> SupercellContainer;

/**
 * @brief We use this test to verify the random access iterator for unnested 
 * array like data structures.
 */
BOOST_AUTO_TEST_CASE(ParticleInFrame)
{
    
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
        
    
    Frame test;
    
    
    auto  childConceptJump1 = hzdr::makeIteratorConcept(
                             hzdr::makeAccessor(),
                             hzdr::makeNavigator(
                                 Offset(0),
                                 Jumpsize(1)));
    

    auto view = makeView(test,
                         childConceptJump1);

    int_fast32_t sum = 0; 
    for(auto it=view.begin(); it!=view.end()-2; it++)
    {
        sum += (*it).data[0] + (*it).data[1];
    }
    // sum [0, 15] = 120
    BOOST_TEST(sum == 120);
    sum = 0; 
    for(auto it=view.begin()+2; it!=view.end()-2; it++)
    {
        sum += (*it).data[0] + (*it).data[1];
    }
    // sum [4, 15] = 114
    BOOST_TEST(sum == 114);
    
    sum = 0;
    for(auto it=view.begin()+2; it<view.end()-2; it+=3)
    {
        sum += (*it).data[0] + (*it).data[1];
    }
    // sum ( (4,5) + (10, 11)) 
    BOOST_TEST(sum == 30);
    
    sum = 0;
    for(auto it=view.rbegin()-1; it>view.rend()+2; it-=3)
    {
        sum += (*it).data[0] + (*it).data[1];
    }
    // sum ( (16, 17) + (10, 11) + (4,5) ) 
    BOOST_TEST(sum == 63);
    
    // check >= <=
    auto it1 = view.begin();
    auto it2 = view.begin();
    
    BOOST_TEST((it1 <= it2));
    BOOST_TEST((it1 >= it2));
    
    BOOST_TEST(not (it1+3 <= it2));
    BOOST_TEST((it1+4 >= it2));
    
    BOOST_TEST((it1 <= it2 + 2));
    BOOST_TEST(not (it1 >= it2 + 8));
    
    // check < >
    BOOST_TEST(not (it1 < it2));
    BOOST_TEST(not (it1 > it2));
    
    BOOST_TEST(not (it1+3 < it2));
    BOOST_TEST((it1+4 > it2));
    
    BOOST_TEST((it1 < it2 + 2));
    BOOST_TEST(not (it1 > it2 + 8));

    
    auto  childConceptJump3 = hzdr::makeIteratorConcept(
                             hzdr::makeAccessor(),
                             hzdr::makeNavigator(
                                 Offset(0),
                                 Jumpsize(3)));

    
    
    // check other jumpsizes
    auto viewJump3 = makeView(
        test,
        childConceptJump3);
    
    BOOST_TEST((view.begin() + 3 == viewJump3.begin() + 1));
    BOOST_TEST((view.begin() + 3 == viewJump3.begin() + 1));

    sum = 0; 
    for(auto it=viewJump3.begin(); it<viewJump3.end(); it++)
    {
        sum += (*it).data[0] + (*it).data[1];
    }
    // 0:(0,1) + 3:(6,7) + 6:(12,13) + 9:(18,19) = 76
    BOOST_TEST(sum == 76);
    sum = 0; 
    for(auto it=viewJump3.begin(); it<viewJump3.end()-2; it++)
    {

        sum += (*it).data[0] + (*it).data[1];
    }
    // 0:(0,1) + 3:(6,7)  = 14
    BOOST_TEST(sum == 14);
    
    sum = 0; 
    for(auto it=viewJump3.begin()+1; it<viewJump3.end()-2; it++)
    {
        sum += (*it).data[0] + (*it).data[1];
    }
    // 3:(6,7) = 13
    BOOST_TEST(sum == 13);
    
}


BOOST_AUTO_TEST_CASE(ParticlInSupercell)
{

    /** We like to test the following things
     * 1. += -= 
     * 2. different offsets and jumpsizes and n
     */
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    auto nbFrames = 5u;
    auto nbParticleInLastFrame = 5u;
    
    Supercell supercell(nbFrames, nbParticleInLastFrame);
    
    std::vector<uint> jumpsizes{1u, 2u, 3u, 4u};
    std::vector<uint> offsets{0u, 1u, 2u, 3u, 4u};
    std::vector<uint> ns{1u, 2u, 3u, 4u};
    std::cout << supercell << std::endl;
   // 1. we change the outer iterator
    for(auto jump : jumpsizes)
        for(auto off : offsets)
            for(auto  n : ns)
            {
                
                auto  childConceptJump1 = hzdr::makeIteratorConcept(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jump)),
                                            hzdr::makeIteratorConcept(
                                                hzdr::makeAccessor(),
                                                hzdr::makeNavigator(
                                                    Offset(0),
                                                    Jumpsize(1))));
                
                // We calc the number of elements, which would be the result
                auto nbFullFrames = (nbFrames - 1u - off + jump - 1u) / jump;
                auto nbParticles = (nbFullFrames * 10u);
                
                // calculate the first index in the last frame
                
                uint first = (n - (nbParticles % n)) % n;
                nbParticles = (nbParticles + n - 1 ) / n;
                if((nbFrames - 1u - off) % jump == 0 and off < nbFrames)
                    // we add the unfull frame
                    for(uint i=first; i<nbParticleInLastFrame; i+= n)
                    {

                        nbParticles++;
                    }
                
                auto view = makeView(supercell,childConceptJump1);
                uint counter = 0u;
                for(auto it=view.begin(); it!=view.end(); it+=n)
                {
                    
                    ++counter;
                }
                BOOST_TEST(counter == nbParticles);
                
                counter = 0u;
                for(auto it=view.rbegin(); it!=view.rend(); it-=n)
                {
                    ++counter;
                }
                BOOST_TEST(counter == nbParticles);
                
            }

   // 2. we change the inner iterator
    for(auto jump : jumpsizes)
        for(auto off : offsets)
            for(auto  n : ns)
            {
                auto  childConceptJump1 = hzdr::makeIteratorConcept(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)),
                                            hzdr::makeIteratorConcept(
                                                hzdr::makeAccessor(),
                                                hzdr::makeNavigator(
                                                    Offset(off),
                                                    Jumpsize(jump))));
                
                // We calc the number of elements, which would be the result
                auto nbParticlesPerFrame = (10u - off + jump - 1u) / jump;
                auto nbParticles = (nbFrames - 1u) * nbParticlesPerFrame;
                
                // calculate the first index in the last frame
                
                
                for(uint i=off; i<nbParticleInLastFrame; i+= jump)
                {
                    nbParticles++;
                }
                nbParticles = (nbParticles + n - 1u) / n;
                
                auto view = makeView(supercell,childConceptJump1);
                uint counter = 0u;
                for(auto it=view.begin(); it!=view.end(); it+=n)
                {
                    ++counter;
                }
                BOOST_TEST(counter == nbParticles);
                
                counter = 0u;
                for(auto it=view.rbegin(); it!=view.rend(); it-=n)
                {

                    ++counter;
                }
                BOOST_TEST(counter == nbParticles);
                
            }

}

BOOST_AUTO_TEST_CASE(ParticleAttrubutesInSupercell)
{
    /** We like to test the following things
     * 1. += -= 
     * 2. different offsets and jumpsizes and n
     */
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    auto nbFrames = 5u;
    auto nbParticleInLastFrame = 5u;
    
    Supercell supercell(nbFrames, nbParticleInLastFrame);
    
    std::vector<uint> jumpsizes{1u, 2u, 3u, 4u};
    std::vector<uint> offsets{0u, 1u, 2u, 3u, 4u};
    std::vector<uint> ns{1u, 2u, 3u};
    std::cout << supercell << std::endl;
   // 1. we change the outer iterator
    for(auto jump : jumpsizes)
        for(auto off : offsets)
            for(auto  n : ns)
            {
                std::cout << "jump " << jump << " off " << off << " n " << n << std::endl;
                auto  childConceptJump1 = hzdr::makeIteratorConcept(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jump)),
                                            hzdr::makeIteratorConcept(
                                                hzdr::makeAccessor(),
                                                hzdr::makeNavigator(
                                                    Offset(0),
                                                    Jumpsize(1)),
                                            hzdr::makeIteratorConcept(
                                                hzdr::makeAccessor(),
                                                hzdr::makeNavigator(
                                                    Offset(0),
                                                    Jumpsize(1)))));
                
                // We calc the number of elements, which would be the result
                auto nbFullFrames = (nbFrames - 1u - off + jump - 1u) / jump;
                auto nbParticles = (nbFullFrames * 20u);
                
                // calculate the first index in the last frame
                
                uint first = (n - (nbParticles % n)) % n;
                nbParticles = (nbParticles + n - 1 ) / n;
                if((nbFrames - 1u - off) % jump == 0 and off < nbFrames)
                    // we add the unfull frame
                    for(uint i=first; i<nbParticleInLastFrame * 2u; i+= n)
                    {

                        nbParticles++;
                    }
                
                auto view = makeView(supercell,childConceptJump1);
                uint counter = 0u;
                for(auto it=view.begin(); it!=view.end(); it+=n)
                {
                    
                    ++counter;
                }
                BOOST_TEST(counter == nbParticles);
                
                counter = 0u;
                for(auto it=view.rbegin(); it!=view.rend(); it-=n)
                {
                    std::cout << *it << std::endl;
                    ++counter;
                }
                BOOST_TEST(counter == nbParticles);
            }
            
}

