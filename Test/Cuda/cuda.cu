
#include "deepiterator/../../Test/Cuda/cuda.hpp"
#include "deepiterator/PIC/SupercellContainerManager.hpp"
#include "deepiterator/PIC/SupercellManager.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"


typedef hzdr::Particle<int32_t, 2> Particle;
typedef hzdr::Frame<Particle, 256> Frame;
typedef hzdr::Supercell<Frame> Supercell;
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
    auto prescription = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0u),
            Jumpsize(1u)),
        hzdr::makeIteratorPrescription(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                Offset(threadIdx.x),
                Jumpsize_256())));
    auto view = hzdr::makeView(*supercell, prescription);
                                            

     for(auto it=view.begin(); it != view.end(); ++it)
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

template< typename T=void>
__device__ 
void 
addParticle(Supercell & supercell, int *value)
{

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    typedef hzdr::SelfValue<uint_fast32_t, 256u> Jumpsize_256;
    auto prescriptionParticle = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0),
            Jumpsize(1)),
        hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(threadIdx.x),
                    Jumpsize_256())));

        // over all frames in a supercell
    auto view= makeView(supercell, prescriptionParticle);

    auto it = view.begin();
    /*
    for(auto it = view.begin(); it != view.end(); ++it)
    {
     //   value[threadIdx.x] += (*it).data[0];
    }
    */
    
}


template< typename T=void>
__global__
void 
addAllParticlesInSupercell(Supercell *supercell, const uint nbSupercells)
{
    const int nbSupercellAdd = 2;
    // define all needed types 
    typedef hzdr::SupercellContainer<Supercell> SupercellContainer;
    typedef typename Supercell::FrameType Frame;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    typedef hzdr::SelfValue<uint_fast32_t, 256u> Jumpsize_256;
    // define shared variables
    
    SupercellContainer supercellContainer(supercell, nbSupercells);  
    auto prescriptionSupercell = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0u),
            Jumpsize(1u)));
            
    auto prescriptionParticle = hzdr::makeIteratorPrescription(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                Offset(0u),
                Jumpsize(1u)),
            hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(threadIdx.x),
                    Jumpsize_256())));
    
    auto viewSupercellContainer = hzdr::makeView(supercellContainer, prescriptionSupercell);
    auto test = viewSupercellContainer.begin();
    
    // Add all the particles where enough supercell are there
    for(auto it=viewSupercellContainer.begin(); it != viewSupercellContainer.end() - (nbSupercellAdd - 1); ++it)
    {
     
        __shared__ int value[256];
        value[threadIdx.x] = 0;
        __syncthreads();
        
        for(int i = 0; i < nbSupercellAdd; ++i)
        {
            addParticle( *(it + i), value);
        }
        /*
        // write back
       
       __shared__ int result;
        result = 0;
        __syncthreads();
        
        atomicAdd(&result, value[threadIdx.x]);
        __syncthreads();
        
       
        auto viewParticle = hzdr::makeView(*it, prescriptionParticle);
        
        for(auto itParticle = viewParticle.begin();
            itParticle != viewParticle.end();
            ++itParticle)
        {
            (*itParticle).data[1] = result;
        }
       */ 
        
        
    }
    /*                
    // add the remaining supercells
    auto counter = (nbSupercellAdd - 1);
    for(auto it=viewSupercellContainer.end() - (nbSupercellAdd - 1); it != viewSupercellContainer.end(); ++it)
    {
        __shared__ int value[256];
        value[threadIdx.x] = 0;
        __syncthreads();
        for(int i = 0; i < counter; ++i)
        {
            addParticle( *(it + i), value);
        }
        --counter;
        // write back
        __shared__ int result;
        result = 0;
        __syncthreads();
        
        atomicAdd(&result, value[threadIdx.x]);
        __syncthreads();
        

        auto viewParticle = hzdr::makeView(*it, prescriptionParticle);
        
        for(auto itParticle = viewParticle.begin();
            itParticle != viewParticle.end();
            ++itParticle)
        {
            (*itParticle).data[1] = result;
        }
        
    }
    */
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

