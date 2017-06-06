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
#include "Traits/HasOffset.hpp"
#include <omp.h>
#include "Iterator/RuntimeTuple.hpp"
#include "Definitions/hdinline.hpp"
#include "Iterator/Collective.hpp"
#include "PIC/SupercellManager.hpp"














template<typename Supercell>
 __global__
 void 
 myKernel(Supercell *supercell, const int nbParticleInLastFrame)
 {   
    typedef typename Supercell::FrameType Frame;
    const int jumpsizeParticle = 256;
    const int offsetParticle = threadIdx.x;
    const int nbElementsParticle = nbParticleInLastFrame;
    typedef hzdr::runtime::TupleFull RuntimeTuple;
    
    const RuntimeTuple runtimeVarParticle(offsetParticle, nbElementsParticle, jumpsizeParticle);
    
    
    const int jumpsizeFrame = 1;
    const int offsetFrame = 0;
    const int nbElementsFrame = 0;
    const RuntimeTuple runtimeFrame(offsetFrame, nbElementsFrame, jumpsizeFrame);
    
    typedef hzdr::View<Frame, hzdr::Direction::Forward,  hzdr::Collectivity::None,RuntimeTuple> ParticleInFrame;
    
    hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::CudaIndexable, RuntimeTuple, ParticleInFrame> view(supercell, runtimeFrame, ParticleInFrame(nullptr, runtimeVarParticle)); 
    
     auto it=view.begin();

     for(auto it=view.begin(); it!=view.end(); ++it)
     {
         if(*it)
         {
             (**it).data[0] += 1;
        }
     }
}




int main(int argc, char **argv) {
/** 1. erstellen eines 2d Arrays auf der GPU. Die zweite Dimension ist dabei 256
 * Elemente groß.
 * 2. Einen Kernel schreiben, der diese Datenstruktur als eingabe parameter nimmt
 * 3. Die Datenstruktur einer Klasse übergeben 
 * 4. Über die datenstruktur iterieren
*/
    typedef hzdr::Particle<int32_t, 2> Particle;
    typedef hzdr::Frame<Particle, 256> Frame;
    typedef hzdr::SuperCell<Frame> Supercell;
    
    SupercellHandle<Supercell> supercellHandler(1, 100);
    std::cout << "Supercell before calcluation" << std::endl;
    std::cout << *(supercellHandler.supercellCPU);
    myKernel<<<1, 256>>>(supercellHandler.supercellGPU, 100);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
     supercellHandler.copyDeviceToHost();
     std::cout << "Supercell after Calculation" << std::endl;
     std::cout << *(supercellHandler.supercellCPU);
    
     
//     Supercell cell(5, 2), cell2(4, 4);
//     const int nbSuperCells = 5;
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
//   //  hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::None, RuntimeTuple,ParticleInFrame> iterSuperCell1(cell, 
//   //                                                                                                                          runtimeSupercell1,
//   //                                                                                                                          ParticleInFrame(nullptr, runtimeVarParticle1)); 
//     
// // create the second iteartor
// 
//     std::cout << "First Supercell " <<std::endl << cell << std::endl;
//     
//     ParticleInFrame view(cell.firstFrame, runtimeVarParticle1);
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
//         hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::None, RuntimeTuple,ParticleInFrame> iterSuperCell(cell, 
//                                                                                                                                 runtimeSupercell2,
//                                                                                                                                 ParticleInFrame(nullptr, runtimeVarParticle2)); 
//         
//         /// @todo doesnt work
//         for(auto it=iterSuperCell.begin(); it != iterSuperCell.end(); ++it)
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
    createSuperCell<<<1, 1>>>();
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
