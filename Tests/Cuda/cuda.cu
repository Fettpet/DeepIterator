#if 0
#include "Tests/Cuda/cuda.hpp"
#include "PIC/SupercellManager.hpp"
#include "PIC/SupercellContainer.hpp"
#include "PIC/SupercellContainerManager.hpp"
#include "Iterator/Policies.hpp"
#include "DeepIterator.hpp"
#include "Iterator/Accessor.hpp"
#undef _OPENMP
#include "View.hpp"
/***************************************************************
 * first Test case: Add a one to all particles first attribute
 * ******************************************************************/
template<typename T=void>
__global__
void 
FrameInSuperCell(Supercell *supercell, const int nbParticleInLastFrame)
{  
   
    typedef typename Supercell::FrameType Frame;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    typedef hzdr::SelfValue<uint_fast32_t, 256> Jumpsize_256;
    auto it=hzdr::makeIterator(
        *supercell, 
        hzdr::makeAccessor(*supercell),
        hzdr::makeNavigator(
            *supercell,
            hzdr::Direction::Forward(),
            Offset(0),
            Jumpsize(1)),
        hzdr::make_child(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                hzdr::Direction::Forward(),
                Offset(threadIdx.x),
                Jumpsize_256())));
                                            

     for(; not it.isAtEnd(); ++it)
     {
        (*it).data[0] += 1;
     }
}

/**
 * @brief 
 */

void
callSupercellAddOne(Supercell** supercell, int Frames, int nbParticleInFrame)
{
    SupercellHandle<Supercell> supercellHandler(Frames, nbParticleInFrame);


    FrameInSuperCell<<<1, 256>>>(supercellHandler.supercellGPU, nbParticleInFrame);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    supercellHandler.copyDeviceToHost();

    *supercell = new Supercell(*(supercellHandler.supercellCPU));

}


/***************************************************************
 * second Test case: Add all Particles of the Supercell together
 * @brief Within this test we do some particle-particle stuff. We use the current 
 * supercell and the next supercell. We read from the data[1] and write in data[0].
 * Our operation is an add. Some wie add all values in the current, and the next 
 * supercell
 * 
 * The particles are stored in frames, which belong to a supercell. We store the 
 * supercells in a supercellcontainer.
 * *************************************************************/

template<typename T=void>
__device__ 
void 
addParticle(Supercell & cur, int *value)
{
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    auto itFrame = makeIterator(
            cur, 
            makeAccessor(cur),
            makeNavigator(
                cur,
                hzdr::Direction::Forward(),
                Offset(0),
                Jumpsize(1)));
        // over all frames in a supercell

    for(;not itFrame.isAtEnd(); ++itFrame)
    {
    //    printf("Inner loop");
            Frame frame =  *itFrame;
            
            __syncthreads();
            auto itParticle = makeIterator(
                *itFrame,
                makeAccessor(*itFrame),
                makeNavigator(
                    *itFrame,
                    hzdr::Direction::Forward(),
                    Offset(threadIdx.x),
                    Jumpsize(256)));
            for(; not itParticle.isAtEnd(); ++itParticle)
            {
                
                value[threadIdx.x] += (*itParticle).data[0];
            }
    }
}


template<typename T=void>
__global__
void 
addAllParticlesInSupercell(Supercell *supercell, const int nbSupercells)
{
    // define all needed types 
    typedef hzdr::SupercellContainer<Supercell> SupercellContainer;
    typedef typename Supercell::FrameType Frame;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    // define shared variables
    
    SupercellContainer supercellContainer(supercell, nbSupercells);  
    
    auto itSupercell = makeIterator(
                supercellContainer, 
                makeAccessor(supercellContainer),
                makeNavigator(
                    supercellContainer,
                    hzdr::Direction::Forward(),
                    Offset(0),
                    Jumpsize(1)));
  //  over all supercells
    while(not itSupercell.isAtEnd())
    {
        Supercell & cur = *itSupercell;
        __shared__ int value[256];
        value[threadIdx.x] = 0;
        __syncthreads();
        addParticle(cur, value);
        // go to the next supercell
        ++itSupercell;
        if(not itSupercell.isAtEnd())
        {
            addParticle(cur, value);
        }
        
        // write back
        __shared__ int result;
        result = 0;
        __syncthreads();
        
        atomicAdd(&result, value[threadIdx.x]);
        __syncthreads();
        auto itParticle = hzdr::makeIterator(
            cur, 
            hzdr::makeAccessor(cur),
            hzdr::makeNavigator(
                cur,
                hzdr::Direction::Forward(),
                Offset(0),
                Jumpsize(1)),
            hzdr::make_child(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    hzdr::Direction::Forward(),
                    Offset(threadIdx.x),
                    Jumpsize(256))));
        for(; not itParticle.isAtEnd(); ++itParticle)
        {
            (*itParticle).data[1] = result;
        }
        
    }
        

}

void callSupercellSquareAdd(Supercell*** superCellContainer, int nbSupercells, std::vector<int> nbFramesSupercell, std::vector<int> nbParticlesInFrame)
{
    SupercellContainerManager<Supercell> supercellHandler(nbSupercells, nbFramesSupercell, nbParticlesInFrame);
    addAllParticlesInSupercell<<<1, 256>>>(supercellHandler.supercellGPU, nbSupercells);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    supercellHandler.copyDeviceToHost();
    
    *superCellContainer = new Supercell*[nbSupercells];
    for(int i=0; i<nbSupercells; ++i)
    {
        (*superCellContainer)[i] = &(supercellHandler.supercellCPU[i]);
    }
    
}
#endif
