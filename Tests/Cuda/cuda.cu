
#include "Tests/Cuda/cuda.hpp"
#include "PIC/SupercellManager.hpp"
#include "PIC/SupercellContainerManager.hpp"
#undef _OPENMP
#include "Iterator/RuntimeTuple.hpp"
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
 * *************************************************************/
template<typename T=void>
__global__
void 
addAllParticlesInSupercell(Supercell *supercell, const int nbSupercells)
{
    __shared__ int32_t mem[256];
    __shared__ int32_t result;
    printf("%f", supercell[0].firstFrame->particles[threadIdx.x].data[0]);
    for(auto i=0; i<nbSupercells-1; ++i)
    {
        typedef typename Supercell::FrameType Frame;
        const int jumpsizeParticle = 256;
        const int offsetParticle = threadIdx.x;
        const int nbElementsParticle = supercell[i].nbParticlesInLastFrame;
        typedef hzdr::runtime::TupleFull RuntimeTuple;
        
        const RuntimeTuple runtimeVarParticle(offsetParticle, nbElementsParticle, jumpsizeParticle);
        
        
        const int jumpsizeFrame = 1;
        const int offsetFrame = 0;
        const int nbElementsFrame = 0;
        const RuntimeTuple runtimeFrame(offsetFrame, nbElementsFrame, jumpsizeFrame);
        
        typedef hzdr::View<Frame, hzdr::Direction::Forward,  hzdr::Collectivity::None,RuntimeTuple> ParticleInFrame;
        
        // get view for both 
        hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::CudaIndexable, RuntimeTuple, ParticleInFrame> view(supercell[i], runtimeFrame, ParticleInFrame(nullptr, runtimeVarParticle)); 
        hzdr::View<Supercell, hzdr::Direction::Forward,  hzdr::Collectivity::CudaIndexable, RuntimeTuple, ParticleInFrame> view2(supercell[i+1], runtimeFrame, ParticleInFrame(nullptr, runtimeVarParticle)); 
        result = 0;
        for(auto it=view.begin(); it!=view.end(); ++it)
        {
            for(auto it2=view2.begin(); it2!=view2.end(); ++it2)
            {
                if(*it2)
                {
                    
                    mem[threadIdx.x] = (**it2).data[0];
                    __syncthreads();
                    atomicAdd(&result, mem[threadIdx.x]);
                }
            }
            if(*it)
            {
                (**it).data[1] = 1;
            }
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
