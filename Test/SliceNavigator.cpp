/**
 * @author Sebastian Hahn
 * @brief The Sliced Navigator is tested in this file. To do this I test the 
 * following things
 * 1. Test a single Layer
 * 1.1 from begining to end with increasing number 
 * 1.2 from end to beginning with decreasing number
 * 2. Test multiple layers 
 * 2.1 navigator-slice 
 * 2.2 slice-navigator
 * 3. Repeat 1 and 2 with rbegin
 * 
 */
#define BOOST_TEST_MODULE Bidirectional
#include <boost/test/included/unit_test.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"

using namespace boost::unit_test;

typedef hzdr::Particle<int_fast32_t, 2u> Particle;
typedef hzdr::Frame<Particle, 10u> Frame;
typedef hzdr::Supercell<Frame> Supercell;
typedef hzdr::SupercellContainer<Supercell> SupercellContainer;

BOOST_AUTO_TEST_CASE(SingleLayer)
{

    Frame container;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    // forward 0
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jumpsize),
                                            hzdr::Slice<hzdr::slice::Distance, 0>()));
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            
            int counter=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                ++counter;
            }
            
            BOOST_TEST(counter == 1);
        }
        
    // forward 1
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jumpsize),
                                            hzdr::Slice<hzdr::slice::Distance,1>()));
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            auto pos = off;
            auto counter = 0;
            while( pos < 10 )
            {
                pos += jumpsize;
                counter++;
            }
            counter = std::min(2, counter);
            
            int counter2=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                counter2++;
            }
            
            BOOST_TEST(counter == counter2);
        }
        
        
            // Backward 0
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jumpsize),
                                            hzdr::Slice<hzdr::slice::IgnoreLastElements, 0>()));
            
           
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            
            int counter=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                ++counter;
            }
            auto pos = off;
            auto counter2 = 0;
            while( pos < 10)
            {
                pos += jumpsize;
                counter2++;
            }
            //counter = std::min(2, counter);
            
            BOOST_TEST(counter == counter2);
        }
        
    // backward 1
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jumpsize),
                                            hzdr::Slice<hzdr::slice::IgnoreLastElements, 1>()));
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
           // std::cout << container << std::endl;
            int counter=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
            //    std::cout << *it << std::endl;
                ++counter;
            }
            auto pos = off;
            auto counter2 = 0;
            while( pos < 10 - 1)
            {
                pos += jumpsize;
                counter2++;
            }
            
            BOOST_TEST(counter == counter2);
        }
        
    // Now the complete thing with rbegin and rend
    // forward 0
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jumpsize),
                                            hzdr::Slice<hzdr::slice::Distance, 0>()));
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
           // std::cout << container << std::endl;
            int counter=0;
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                //std::cout << *it << std::endl;
                ++counter;
            }

            //counter = std::min(2, counter);
            
            BOOST_TEST(counter == 1);
        }
    // forward 1
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jumpsize),
                                            hzdr::Slice<hzdr::slice::Distance,1>()));
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int counter=0;
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                ++counter;
            }
            
            int counter2 = 0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                ++counter2;
            }
            
            BOOST_TEST(counter == counter2);
        }
        
    // Backward 0
        for(int off=0; off<9; ++off)
            for(int jumpsize=1; jumpsize<9; ++jumpsize)
            {
                auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize),
                                                hzdr::Slice<hzdr::slice::IgnoreLastElements, 0>()));
                
                
                auto view = makeView(
                    container, 
                    childPrescriptionJump1
                );
                int counter=0;
                for(auto it=view.rbegin(); it!=view.rend(); --it)
                {
                    ++counter;
                }
                auto counter2 = 0;
                for(auto it=view.begin(); it!=view.end(); ++it)
                {
                    ++counter2;
                }


                
                BOOST_TEST(counter == counter2);
            }
        
    // forward
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(off),
                                            Jumpsize(jumpsize),
                                            hzdr::Slice<hzdr::slice::IgnoreLastElements, 1>()));
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
           // std::cout << container << std::endl;
            int counter=0;
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                std::cout << "Backward" << *it << std::endl;
                ++counter;
            }
            auto counter2 = 0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                std::cout << "forwrad" << *it << std::endl;
                ++counter2;
            }

            BOOST_TEST(counter == counter2);
        }

}

/**
 * @brief This test is used to check whether the slice naviagatror and the 
 * navigator are compatile to each other
 */
BOOST_AUTO_TEST_CASE(TWOLAYER)
{
    Frame container;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    // forward 0
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1),
                                            hzdr::Slice<hzdr::slice::Distance,1>()
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize)
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int sum=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                sum += *it;
            }
            
            int checksum=0;
            for(int i=0; i<2; ++i)
            {
                for( int j=off; j<2; j+=jumpsize)
                {
                    checksum += container[i][j];
                }
            }
            
            BOOST_TEST(sum == checksum);
        }
        
    for(int off=0; off<8; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1),
                                            hzdr::Slice<hzdr::slice::Distance,1>()
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize)
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );

            std::cout << container << std::endl;
            int sum=0;
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                sum += *it;
            }
            
            int checksum=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                checksum+= *it;
            }

            BOOST_TEST(sum == checksum);
        }

    // backward
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1),
                                            hzdr::Slice<hzdr::slice::Distance,1>()
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize)
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int sum=0;
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                sum += *it;
            }
            
            int checksum=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                checksum+= *it;
            }

            BOOST_TEST(checksum == sum);
        }
        
      
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize),
                                                hzdr::Slice<hzdr::slice::Distance, 0>()
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int sum=0;
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                sum += *it;

            }
            
            int checksum=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                checksum+= *it;
            }
            BOOST_TEST(checksum == sum);
        }
  
    for(int off=0; off<1; ++off)
        for(int jumpsize=1; jumpsize<2; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize),
                                                hzdr::Slice<hzdr::slice::IgnoreLastElements, 0>()
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int sum=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                sum += *it;
            }
            
            int checksum=0;
            for(int i=0; i<10; ++i)
            {
                for( int j=0; j<2; ++j)
                    checksum += container[i][j];
            }
            BOOST_TEST(sum == checksum);
        }
        
    for(int off=0; off<1; ++off)
        for(int jumpsize=1; jumpsize<2; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize),
                                                hzdr::Slice<hzdr::slice::IgnoreLastElements, 1>()
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int sum=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                sum += *it;
            }
            
            int checksum=0;
            for(int i=0; i<10; ++i)
            {
                checksum += container[i][0];
            }

            BOOST_TEST(checksum == sum);
        }
        
    // forward
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize),
                                                hzdr::Slice<hzdr::slice::IgnoreLastElements, 1>()
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int sum=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                sum += *it;
            }
            
            int checksum=0;
            if(off == 0)
                for(int i=0; i<10; ++i)
                {
                    checksum += container[i][0];
                }
                
                BOOST_TEST(checksum == sum);
        }
        
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize),
                                                hzdr::Slice<hzdr::slice::IgnoreLastElements, 1>()
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int sum=0;
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                sum += *it;
            }
            
            int checksum=0;
            for(auto it=view.begin(); it!=view.end(); ++it)
            {
                checksum+= *it;
            }
            BOOST_TEST(checksum == sum);
        }

}

