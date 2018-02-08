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
    typedef hzdr::Particle<int32_t, 1u> Particle;
    typedef hzdr::Frame<Particle, 10u> Frame;
    typedef hzdr::Supercell<Frame> Supercell;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    typedef hzdr::SelfValue<uint_fast32_t, 256u> Jumpsize_256;
    std::cout << std::endl << " Test with runtime jumpsize and offset. We use list<vector<int> >" << std::endl;
    
    auto nbFrames = 2u;
    auto nbParticleInLastFrame = 5u;
    
    Supercell supercell(nbFrames, nbParticleInLastFrame);
    
    std::vector<uint> jumpsizes{1u, 2u, 3u, 4u};
    std::vector<uint> offsets{0u, 1u, 2u, 3u, 4u};
    std::vector<uint> ns{1u, 2u, 3u, 4u};
    std::cout << supercell << std::endl;
    
                auto  childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)),
                                            hzdr::makeIteratorPrescription(
                                                hzdr::makeAccessor(),
                                                hzdr::makeNavigator(
                                                    Offset(3),
                                                    Jumpsize(1))));
                
                // We calc the number of elements, which would be the result
                auto nbParticlesPerFrame = (10u - 3 + 1 - 1u) / 1;
                auto nbParticles = (nbFrames - 1u) * nbParticlesPerFrame;
                
                // calculate the first index in the last frame
                
                
                for(uint i=3; i<nbParticleInLastFrame; i+= 1)
                {
                    nbParticles++;
                }
                nbParticles = (nbParticles + 3 - 1u) / 3;
                
                auto view = makeView(supercell,childPrescriptionJump1);
                uint counter = 0u;
                for(auto it=view.begin(); it!=view.end(); it+=3)
                {
                    std::cout << *it << std::endl;
                    ++counter;
                }
                std::cout << counter << ", " << nbParticles << std::endl;
                
                counter = 0u;
                for(auto it=view.rbegin(); it!=view.rend(); it-=3)
                {

                    ++counter;
                }
                std::cout << counter << ", " << nbParticles << std::endl;
/*    std::cout << testFrame << std::endl;
    std::cout << "Last Element:" ;
    std::cout << *(view.rbegin()) << std::endl;;
    std::cout <<  *(--(++(view.rbegin()))) << std::endl; 
    
    std::cout << "First:" ;
    std::cout << *(view.begin()) << std::endl;;
    std::cout <<  (--(view.begin()) == view.rend()) << std::endl; 
    // 1. Test we iterate over the frame and read all particles 
    auto counter = 0u*/;
 
   
//     for(auto && it=view.begin(); it!= view.end(); it += 3)
//     {
//         if(++counter > 50) 
//             break;
//         std::cout << *it << std::endl;
//     }
    /*
    auto  childPrescriptionOffset2 = hzdr::makeIteratorPrescription(
                                hzdr::makeAccessor(),
                                hzdr::makeNavigator(
                                    Offset(0),
                                    Jumpsize(1)),
                                hzdr::makeIteratorPrescription(
                                    hzdr::makeAccessor(),
                                    hzdr::makeNavigator(
                                        Offset(1),
                                        Jumpsize(1))));
    
    auto view2 = makeView(test,
                         childPrescriptionOffset2);
    for(auto && it=view2.begin(); it!= view2.end(); it += 3)
    {
        if(++counter > 50) 
            break;
        std::cout << (*it)<< std::endl;
    }
    counter = 0;
    /*
    for(auto && it=view2.rbegin(); it!= view2.rend(); it-=2)
    {
   //     std::cout << "index: " << it.index << std::endl;
        if(++counter > 20) 
            break;
        std::cout << *it << std::endl;
    }
    
    
    
  //  auto navigator = makeNavigator(test, FirstElem(), NextElem(),  LastElem());
                    
    
    
//         for(auto it=nbFrames.begin(); it!=nbFrames.end()-1; ++it)
//         {
//             std::cout << *it << std::endl;
//         }
        // first test the iterator
//         for(auto && iterator = hzdr::makeIterator(
//                                     particle, 
//                                     hzdr::makeAccessor(particle),
//                                     hzdr::makeNavigator(particle, 
//                                                      hzdr::Direction::Forward(),
//                                                      offset, 
//                                                      Jumpsize(1)) ); 
//             not iterator.isAtEnd(); 
//             ++iterator)
//         {
//             ++counter;
//         }
//         std::cout << "We count " << counter << ", should be 2" << std::endl;
//         counter = 0;
//         Frame frame(100);
//         std::vector<std::string> test;
//         
// 
//         auto && iter = hzdr::makeIterator(frame,
//                             hzdr::makeAccessor(frame), 
//                             hzdr::makeNavigator(frame,
//                                                 hzdr::Direction::Forward(),
//                                                 Offset(0),
//                                                 Jumpsize(1)),
//                             hzdr::make_child(hzdr::makeAccessor(),
//                                              hzdr::makeNavigator(hzdr::Direction::Forward(),
//                                                                  Offset(0),
//                                                                  Jumpsize(1))));
//                                                 
//         for(; not iter.isAtEnd(); ++iter)
//         {
//   //           std::cout << "size Frame" <<frame.nbParticlesInFrame << std::endl;
//              std::cout << *iter << std::endl;
//             counter ++;
//             if(counter > 205) break;
//         }
//         std::cout << "Counter" << counter << std::endl;
// //         
        
        
        

/** 1. erstellen eines 2d Arrays auf der GPU. Die zweite Dimension ist dabei 256
 * Elemente groß.
 * 2. Einen Kernel schreiben, der diese Datenstruktur als eingabe parameter nimmt
 * 3. Die Datenstruktur einer Klasse übergeben 
 * 4. Über die datenstruktur iterieren
*/
//     typedef hzdr::Particle<int32_t, 2> Particle;
//     typedef hzdr::Frame<Particle, 256> Frame;
//     typedef hzdr::Supercell<Frame> Supercell;
//     
//     SupercellHandle<Supercell> supercellHandler(1, 100);
//     std::cout << "Supercell before calcluation" << std::endl;
//     std::cout << *(supercellHandler.supercellCPU);
//     myKernel<<<1, 256>>>(supercellHandler.supercellGPU, 100);
//     gpuErrchk( cudaDeviceSynchronize() );
//     gpuErrchk( cudaPeekAtLastError() );
//      supercellHandler.copyDeviceToHost();
//      std::cout << "Supercell after Calculation" << std::endl;
//      std::cout << *(supercellHandler.supercellCPU);
    
     
//     Supercell cell(5, 2), cell2(4, 4);
//     const int nbSupercells = 5;
//     const int nbFramesInSupercell = 2;
// 
//     
//     typedef hzdr::View<Frame, hzdr::Direction::Forward,  hzdr::Collectivity::None,RuntimeTuple> ParticleInFrame;
//     
// // create the first iterator
//     const int jumpsizeFrame1 = 1;
//     const int offsetFrame1 = 0;
//     const int nbElementsFrame1 = -1;
//     
//     const RuntimeTuple runtimeSupercell1(offsetFrame1, nbElementsFrame1, jumpsizeFrame1);
//     
//     const int jumpsizeParticle1 = 1;
//     const int offsetParticle1 = 0;
//     const int nbElementsParticle1 = cell.nbParticlesInLastFrame;
//     
//     const RuntimeTuple runtimeVarParticle1(offsetParticle1, nbElementsParticle1, jumpsizeParticle1);
//     
//   //  hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::None, RuntimeTuple,ParticleInFrame> iterSupercell1(cell, 
//   //                                                                                                                          runtimeSupercell1,
//   //                                                                                                                          ParticleInFrame(nullptr, runtimeVarParticle1)); 
//     
// // create the second iteartor
// 
//     std::cout << "First Supercell " <<std::endl << cell << std::endl;
//     
//     ParticleInFrame view(cell.first, runtimeVarParticle1);
//     
//    // view.begin();
//    // view.end();
//     
//     for(auto it=view.begin(); it!=view.end();++it)
//     {
//         if(*it)
//         {
//             std::cout << **it << std::endl;
//         }
//     }
//     
//   //  std::cout << "Second Supercell " << cell2 << std::endl;
// // first add all 
//     const uint nbThreads = 5;
//     int count = 0;
//     const int i=0;
// //     for(int i=0; i<5; ++i)
// //     {   
//         const int jumpsizeFrame2 = 1;
//         const int offsetFrame2 = 0;
//         const int nbElementsFrame2 = 0;
//         
//         const RuntimeTuple runtimeSupercell2(offsetFrame2, nbElementsFrame2, jumpsizeFrame2);
//         
//         const int jumpsizeParticle2 = nbThreads;
//         const int offsetParticle2 = i;
//         const int nbElementsParticle2 = cell.nbParticlesInLastFrame;
//         
//         const RuntimeTuple runtimeVarParticle2(offsetParticle2, nbElementsParticle2, jumpsizeParticle2);
//         
//         hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::None, RuntimeTuple,ParticleInFrame> iterSupercell(cell, 
//                                                                                                                                 runtimeSupercell2,
//                                                                                                                                 ParticleInFrame(nullptr, runtimeVarParticle2)); 
//         
//         /// @todo doesnt work
//         for(auto it=iterSupercell.begin(); it != iterSupercell.end(); ++it)
//         {
//             if(*it)
//             {
//                 (**it).data[0] = i;
//             }
//             else 
//             {
//                
//             }
//              ++count;
//         }   
// //     }
//     std::cout << "Number of invalids " << count << std::endl;
//     std::cout << "First Supercell after Calc" << cell << std::endl;
// 
//    

/*
    const int dim = 2;
    int *array_h;
    int *array_d;
    
    array_h = new int[dim*256];
    
    for(int i=0; i< dim*256; ++i)
    {
        array_h[i] = 1;
    }
    
    gpuErrchk(cudaMalloc(&array_d, sizeof(int) * dim * 256));
    gpuErrchk(cudaMemcpy(array_d, array_h, sizeof(int) * dim * 256, cudaMemcpyHostToDevice));
    createSupercell<<<1, 1>>>();
    myKernel<<<1, 256>>>();
    
    std::cout << "It works" << std::endl;
    
    gpuErrchk(cudaMemcpy(array_h, array_d, sizeof(int) * dim * 256, cudaMemcpyDeviceToHost));
    
    for(int i=0; i<dim; ++i)
    {
        for(int j=0; j<256; ++j)
        {
            std::cout << array_h[j + i*256] << " ";
        }
        std::cout << std::endl;
    }
    

    delete [] array_h;
    return EXIT_SUCCESS;
    */
    
}
