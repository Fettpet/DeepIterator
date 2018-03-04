/**
 * @brief Within this test we test the runtime of our iterator, compared with the
 * std iterator. We think that your approach is slower since the std iterator is
 * more speciallized. We test different sizes: 1M, 10M, 100M, 1G. We use 
 * different Tests:
 * 1. with compiletime offset, Jumpsize
 * 2. with runtime offset, jumpsize
 * A. list< vector< float > >
 * B. list< float >  
 */


#define BOOST_TEST_MODULE ForwardIterator
#include <boost/test/included/unit_test.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"

#include <boost/timer.hpp>

uint const nbTries = 25u;
using namespace boost::unit_test;
typedef hzdr::Particle<int_fast32_t, 1u> Particle;
typedef hzdr::Frame<Particle, 100u> Frame;
typedef hzdr::Supercell<Frame> Supercell;
typedef hzdr::SupercellContainer<Supercell> SupercellContainer;

typedef hzdr::Frame<Particle, 1u> FrameSingle;
typedef hzdr::Supercell<FrameSingle> SupercellSingle;


BOOST_AUTO_TEST_CASE(Test1B)
{
    std::cout << std::endl <<  "Second Test with compiletime jumpsize and offset. we use list<int>" << std::endl;
    typedef hzdr::SelfValue<uint_fast32_t, 0u> Offset;
    typedef hzdr::SelfValue<uint_fast32_t, 1u> Jumpsize;
    
    std::vector<uint> sizes{1000000, 10000000, 100000000};
    
    for(auto const nbElem: sizes)
    {
        double time1Layer(0), timetrival(0);
        for(auto i=0u; i<nbTries; ++i)
        {
            auto && data = SupercellSingle(nbElem, 1u);
            boost::timer timer;
            uint_fast64_t sum = static_cast<uint_fast64_t>(0);
            timer.restart();
            auto buffer = data.firstFrame;
            while(buffer != nullptr)
            {
                sum += buffer->operator[](0).data[0];
                buffer = buffer->nextFrame;
            }
            timetrival += timer.elapsed();
            
            
            auto && concept1Layer = hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(),
                    Jumpsize()
                )
            );
            auto && view1Layer = hzdr::makeView(
                data, 
                concept1Layer
            );
            timer.restart();
            uint_fast64_t sum1Layer = static_cast<uint_fast64_t>(0);
            for(auto && it=view1Layer.begin(); it != view1Layer.end(); ++it)
            {
                sum1Layer += (*it)[0].data[0];
            }
            time1Layer += timer.elapsed();
            

            BOOST_TEST(sum ==sum1Layer);

        }
        std::cout << "After " << nbTries << " tries:" << std::endl;
        std::cout << "Needed time for " << nbElem << " elements with trival access: " << (timetrival / (double)nbTries)<< std::endl;
        std::cout << "Needed time for " << nbElem << " elements with one layer iterator access: " << (time1Layer / (double)nbTries) << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(Test2B)
{
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    
    std::vector<uint> sizes{1000000, 10000000, 100000000};
    std::cout << std::endl << "Test with runtime offset and jumpsize. We use list<int>" << std::endl;
    for(auto const nbElem: sizes)
    {
        double time1Layer(0), timetrival(0);
        for(auto i=0u; i<nbTries; ++i)
        {
            auto && data = SupercellSingle(nbElem, 1u);
            boost::timer timer;
            uint_fast64_t sum = static_cast<uint_fast64_t>(0);
            timer.restart();
            auto buffer = data.firstFrame;
            while(buffer != nullptr)
            {
                sum += buffer->operator[](0).data[0];
                buffer = buffer->nextFrame;
            }
            timetrival += timer.elapsed();
            
            
            auto && concept1Layer = hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(0u),
                    Jumpsize(1u)));
            auto && view1Layer = hzdr::makeView(data, concept1Layer);
            timer.restart();
            uint_fast64_t sum1Layer = static_cast<uint_fast64_t>(0);
            for(auto && it=view1Layer.begin(); it != view1Layer.end(); ++it)
            {
                sum1Layer += (*it)[0].data[0];
            }
            time1Layer += timer.elapsed();
            
            

            BOOST_TEST(sum ==sum1Layer);
        }
        std::cout << "After " << nbTries << " tries:" << std::endl;
        std::cout << "Needed time for " << nbElem << " elements with one layer access: " << (time1Layer / (double)nbTries) << std::endl;
        std::cout << "Needed time for " << nbElem << " elements with trival access: " << (timetrival / (double)nbTries)<< std::endl;
    }
    

}








BOOST_AUTO_TEST_CASE(Test2A)
{
    std::cout << std::endl << " Test with runtime jumpsize and offset. We use list<vector<int> >" << std::endl;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
   
    
    std::vector<uint> sizes{10000, 100000, 1000000};
    
    for(auto const nbElem: sizes)
    {
        double time1Layer(0), timetrival(0);
        for(auto i=0u; i<nbTries; ++i)
        {
            auto && data = Supercell(nbElem, 100u);
            boost::timer timer;
            uint_fast64_t sum = static_cast<uint_fast64_t>(0);
            timer.restart();
            auto buffer = data.firstFrame;
            while(buffer != nullptr)
            {
                
                for(int i=0;i <100; ++i)
                {
                    sum += buffer->operator[](i).data[0];
                }
                buffer = buffer->nextFrame;
            }
            timetrival += timer.elapsed();
            
            
            auto && concept1Layer = hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(0u),
                    Jumpsize(1u)), 
                hzdr::makeIteratorPrescription(
                    hzdr::makeAccessor(),
                    hzdr::makeNavigator(
                        Offset(0u),
                        Jumpsize(1u))));
            
            auto && view1Layer = hzdr::makeView(data, concept1Layer);
            timer.restart();
            uint_fast64_t sum1Layer = static_cast<uint_fast64_t>(0);
            for(auto && it=view1Layer.begin(); it != view1Layer.end(); ++it)
            {
                sum1Layer += (*it).data[0];
            }
            time1Layer += timer.elapsed();
            
            

            BOOST_TEST(sum ==sum1Layer);

        }
        std::cout << "After " << nbTries << " tries:" << std::endl;
        std::cout << "Needed time for " << nbElem << " elements with trival access: " << (timetrival / (double)nbTries)<< std::endl;
        std::cout << "Needed time for " << nbElem << " elements with one layer iterator access: " << (time1Layer / (double)nbTries) << std::endl;
    }
}



BOOST_AUTO_TEST_CASE(Test1A)
{
    std::cout << std::endl << "Test with compiletime jumpsize and offset. We use list<vector<int> >" << std::endl;
    typedef hzdr::SelfValue<uint_fast32_t, 0u> Offset;
    typedef hzdr::SelfValue<uint_fast32_t, 1u> Jumpsize;
    
    std::vector<uint> sizes{10000, 100000, 1000000};
    
    for(auto const nbElem: sizes)
    {
        double time1Layer(0), timetrival(0);
        for(auto i=0u; i<nbTries; ++i)
        {
            auto && data = Supercell(nbElem, 100u);
            boost::timer timer;
            uint_fast64_t sum = static_cast<uint_fast64_t>(0);
            timer.restart();
            auto buffer = data.firstFrame;
            while(buffer != nullptr)
            {
                for(int i=0;i <100; ++i)
                {
                    sum += buffer->operator[](i).data[0];
                }
                buffer = buffer->nextFrame;
            }
            timetrival += timer.elapsed();
            
            
            auto && concept1Layer = hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(),
                    Jumpsize()), 
                hzdr::makeIteratorPrescription(
                    hzdr::makeAccessor(),
                    hzdr::makeNavigator(
                        Offset(),
                        Jumpsize())));
            
            auto && view1Layer = hzdr::makeView(data, concept1Layer);
            timer.restart();
            uint_fast64_t sum1Layer = static_cast<uint_fast64_t>(0);
            for(auto && it=view1Layer.begin(); it != view1Layer.end(); ++it)
            {
                sum1Layer += (*it).data[0];
            }
            time1Layer += timer.elapsed();
            
            

            BOOST_TEST(sum ==sum1Layer);

        }
        std::cout << "After " << nbTries << " tries:" << std::endl;
        std::cout << "Needed time for " << nbElem << " elements with trival access: " << (timetrival / (double)nbTries)<< std::endl;
        std::cout << "Needed time for " << nbElem << " elements with one layer iterator access: " << (time1Layer / (double)nbTries) << std::endl;
    }
}


