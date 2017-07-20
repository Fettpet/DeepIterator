#include "Tests/Cuda/cuda.hpp"
#include "PIC/SupercellManager.hpp"
#include "PIC/SupercellContainer.hpp"
#include "PIC/SupercellContainerManager.hpp"
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

    
    
    

    const auto offset = threadIdx.x;
    typedef hzdr::View<Frame, hzdr::Direction::Forward<256>> ParticleInFrame;
    hzdr::View<Supercell, hzdr::Direction::Forward<1>, ParticleInFrame> view(supercell, ParticleInFrame(offset)); 
    auto it=view.begin();
    printf("Offset %i\n", it.childIter.index);
     for(; it!=view.end(); ++it)
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
    // define all needed types 
    typedef hzdr::SupercellContainer<Supercell> SupercellContainer;
    typedef typename Supercell::FrameType Frame;
    typedef hzdr::View<SupercellContainer, 
                       hzdr::Direction::Forward<1> > ViewSupercellContainer;
    typedef hzdr::View<Frame, 
                       hzdr::Direction::Forward<1> > ParticleInFrame;
    typedef  hzdr::View<Supercell,
                        hzdr::Direction::Forward<256>,  
                        ParticleInFrame> ParticleInSupercellView;
    // define shared variables
    __shared__ int32_t mem[256];
    __shared__ int32_t result;
    
    // create the iteratable container.
    SupercellContainer supercellContainer(supercell, nbSupercells);  
// 
    // create the first second: over all supercells
    ViewSupercellContainer viewSupercellContainer(supercellContainer);
 
    for(auto itSupercell=viewSupercellContainer.begin();
        itSupercell != viewSupercellContainer.end();
        ++itSupercell)
    {
        // Add all particles within the first supercell
        result = 0;
        mem[threadIdx.x] = 0;
        if(*itSupercell)
        {
            ParticleInSupercellView viewSupercell(**itSupercell);
            
            for(auto itParticle = viewSupercell.begin();
                    itParticle != viewSupercell.end();
                    ++itParticle)
            {
                if(*itParticle)
                    mem[threadIdx.x] += (**itParticle).data[0];
            }
            
        }
        // add all particles within the second supercell
        auto nextSupercell = itSupercell;
        ++nextSupercell;
        if(*(nextSupercell))
        {
            ParticleInSupercellView viewSupercell(**(nextSupercell));
            
            for(auto itParticle = viewSupercell.begin();
                    itParticle != viewSupercell.end();
                    ++itParticle)
            {
                if(*itParticle)
                    mem[threadIdx.x] += (**itParticle).data[0];
            }
            
        }
        // write the results back
        atomicAdd(&result, mem[threadIdx.x]);
        __syncthreads();
        
                if(*itSupercell)
        {
            ParticleInSupercellView viewSupercell(**itSupercell);
            
            for(auto itParticle = viewSupercell.begin();
                    itParticle != viewSupercell.end();
                    ++itParticle)
            {
                if(*itParticle)
                    (**itParticle).data[1] = result;
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
