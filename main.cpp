
/** \mainpage 
 * # Motivation {#section1}
 * The open source project 
 * <a href="https://github.com/ComputationalRadiationPhysics/picongpu/">
 * PIConGPU </a> has several datastructures. The particles have some attributes 
 * like position and speed. The particles are grouped in frames. A frame has a 
 * maximum number of particles within the frame. The frames are part of supercell.
 * A supercell contains a double linked list of frames. Each frame within a 
 * supercell, except the last one, has the maximum number of particles. A supercell
 * is devided in several cells. The affiliation of a particle to a cell is a 
 * property of the particle.
 * The goal of this project is to write an iterator that can go through the data 
 * structes. It should be possible to go over nested datastructers. For example 
 * over all particles within a supercell. We would use the iterator on GPU and 
 * CPU. So it should be possible to overjump some elements, such that no element 
 * is used more than ones, in a parallel application.
 * 
 * # The DeepIterator {#section2}
 * The DeepIterator class is used to iterator over interleaved data 
 * structures. The simplest example is for an interleaved data structure is 
 * std::vector< std::vector< int > >. The deepiterator iterates over all ints 
 * within the structure. For more details see DeepIterator and View. 
 * \see DeepIterator.hpp \see View.hpp
 * 
 * # Changes in the datastructure {#section3}
 * The number of elements in the last frame was a property of the supercell. This
 * is a problem. To illustrate this, we give an example. We like to iterate 
 * over all particles in all Supercells. Your first attempt was a runtime variable.
 * The user gives the size of the last frame explicitly when the view is created.
 * This also requires a function which decide wheter the last frame is reached.
 * The problem is, if we go over more than on supercell the number of particles 
 * within the last frame changed with each supercell. Our first approach cann't 
 * handle this case. To overcame this, we would need a function that gives the 
 * size of the last frame. But this information is hidden two layers above. This
 * doesnt see like a good design. 
 * 
 * So we decide to change the datastructres of PIConGPU. Each frame has a fixed 
 * size, i.e. how many particle are at most in the frame. We give the frame a 
 * variable numberParticles. This is the number of particles within the frame.
 * The number must be smaller than the maximum number of particles.
 */

#include <iostream>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <typeinfo>
#include <memory>
#include <cstdlib>
#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "View.hpp"
#include <omp.h>
#include "Definitions/hdinline.hpp"


#include "Tests/Cuda/cuda.hpp"
#include "DeepIterator.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Prescription.hpp"

#include "Traits/NumberElements.hpp"
#include "Definitions/hdinline.hpp"
#include <boost/timer.hpp>




int main(int , char **) {
    typedef hzdr::Particle<int32_t, 2u> Particle;
    typedef hzdr::Frame<Particle, 10u> Frame;
    typedef hzdr::Supercell<Frame> Supercell;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    typedef hzdr::SelfValue<uint_fast32_t, 256u> Jumpsize_256;
  
    Frame container;    
    std::cout << container << std::endl;
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
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                sum += *it;
                std::cout << *it << std::endl;
            }
            
            auto idx = 0;
            if(off == 0 and jumpsize == 1)
                idx = 1;
            int checksum=0;
            if(off == 0)
                for(int i=0; i<10; ++i)
                {
                    checksum += container[i][idx];
                }
            std::cout << off << " " << jumpsize << " " << sum << " " << checksum << std::endl;
        }

}
