
#define BOOST_TEST_MODULE CollectiveOpenMP
#include <boost/test/included/unit_test.hpp>
#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "DeepIterator.hpp"
#include "View.hpp"
#include "Iterator/RuntimeTuple.hpp"
#include <omp.h>
using namespace boost::unit_test;
typedef hzdr::Particle<int_fast32_t, 2> Particle;
typedef hzdr::Frame<Particle, 10> Frame;
typedef hzdr::SuperCell<Frame> Supercell;
typedef hzdr::runtime::TupleFull RuntimeTuple;

#if 0
BOOST_AUTO_TEST_CASE(PositionsInFrame)
{
    Frame test;
    
    typedef hzdr::View<Frame, 
                       hzdr::Direction::Forward, 
                       hzdr::Collectivity::OpenMPIndexable, 
                       hzdr::runtime::TupleOpenMP> ParticleInFrame;
   
    hzdr::runtime::TupleOpenMP runtimeFrame(10);
    
    ParticleInFrame view(test, runtimeFrame);
    int_fast32_t sum(0);
// first test:  sum all values (should be 90)

#pragma omp parallel shared(test) reduction(+:sum) 
    {       
        int count=0;
        for(auto it = view.begin(); it!=view.end(); ++it)
        {
            count++;
            if(*it)
            {
                sum +=       (**it).data[0];
            }
                

        }

    }
    
    BOOST_TEST(sum == 90);

    sum = 0;
// second test: We sum only the first and search the max (
    #pragma omp parallel reduction(max:sum)
    {       
        sum = 0;
        for(auto it = view.begin(); it!=view.end(); ++it)
        {

            if(*it)
            {
                sum +=       (**it).data[0];

            }
                

        }
    }
    auto nbThreads = omp_get_max_threads();
    int nbElem{9};
    int_fast32_t sumControl{0};
    while(nbElem >= 0)
    {
        sumControl += test[nbElem].data[0];
        nbElem -= nbThreads;
    }
    BOOST_TEST(sum == sumControl);

// Test backward. must be the same result as forward
        typedef hzdr::View<Frame, 
                       hzdr::Direction::Backward, 
                       hzdr::Collectivity::OpenMPIndexable, 
                       hzdr::runtime::TupleOpenMP> ParticleInFrameBackward;
                       
    ParticleInFrameBackward view2(test, runtimeFrame);
    sum = 0;

// first test:  sum all values (should be 90
#pragma omp parallel reduction(max:sum)
    {       
        sum = 0;


        for(auto it = view2.begin(); it!=view2.end(); ++it)
        {

            if(*it)
            {
                sum +=       (**it).data[0];
            }
                

        }

    }
    BOOST_TEST(sum == sumControl);


}

BOOST_AUTO_TEST_CASE(FramesInSuperCells)
{
    Supercell test(10, 2);
     
    typedef hzdr::View<Supercell, 
                       hzdr::Direction::Forward, 
                       hzdr::Collectivity::OpenMPIndexable, 
                       hzdr::runtime::TupleOpenMP> FrameInSupercell;
    hzdr::runtime::TupleOpenMP tuple(10);
    
    FrameInSupercell view(test, tuple);
    int sum=0;
#pragma omp parallel reduction(+:sum)
    {
        sum=0;
        int iter = 0;
        for(auto it = view.begin(); it != view.end(); ++it)
        {
            if(*it)
            {
                sum += (**it).sum();
            }
            iter++;
        }
#pragma omp critical
        std::cout << "I'm " << omp_get_thread_num() << " and I need " << iter << " iterations" << std::endl;
    }
    std::cout << "Summe: " << sum << std::endl;
                       
  
}
#endif