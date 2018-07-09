/* Copyright 2018 Sebastian Hahn

 * This file is part of DeepIterator.
 *
 * DeepIterator is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DeepIterator is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DeepIterator.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "deepiterator/../../Test/Cuda/cuda.hpp"
#include "deepiterator/PIC/SupercellContainerManager.hpp"
#include "deepiterator/PIC/SupercellManager.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"


typedef deepiterator::Particle<int32_t, 2> Particle;
typedef deepiterator::Frame<Particle, 256> Frame;
typedef deepiterator::Supercell<Frame> Supercell;
/***************************************************************
 * first Test case: Add a one to all particles first attribute
 * ******************************************************************/
template<typename T=void>
__global__
void 
FrameInSuperCell(Supercell *supercell, const int nbParticleInLastFrame)
{
   
    typedef typename Supercell::FrameType Frame;
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    typedef deepiterator::SelfValue<uint_fast32_t, 256> Jumpsize_256;
    auto prescription = deepiterator::makeIteratorPrescription(
        deepiterator::makeAccessor(),
        deepiterator::makeNavigator(
            Offset(0u),
            Jumpsize(1u)),
        deepiterator::makeIteratorPrescription(
            deepiterator::makeAccessor(),
            deepiterator::makeNavigator(
                Offset(threadIdx.x),
                Jumpsize_256())));
    auto view = deepiterator::makeView(*supercell, prescription);
                                            

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

    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    typedef deepiterator::SelfValue<uint_fast32_t, 256u> Jumpsize_256;
    auto prescriptionParticle = deepiterator::makeIteratorPrescription(
        deepiterator::makeAccessor(),
        deepiterator::makeNavigator(
            Offset(0),
            Jumpsize(1)),
        deepiterator::makeIteratorPrescription(
                deepiterator::makeAccessor(),
                deepiterator::makeNavigator(
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
    typedef deepiterator::SupercellContainer<Supercell> SupercellContainer;
    typedef typename Supercell::FrameType Frame;
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    typedef deepiterator::SelfValue<uint_fast32_t, 256u> Jumpsize_256;
    // define shared variables
    
    SupercellContainer supercellContainer(supercell, nbSupercells);  
    auto prescriptionSupercell = deepiterator::makeIteratorPrescription(
        deepiterator::makeAccessor(),
        deepiterator::makeNavigator(
            Offset(0u),
            Jumpsize(1u)));
            
    auto prescriptionParticle = deepiterator::makeIteratorPrescription(
            deepiterator::makeAccessor(),
            deepiterator::makeNavigator(
                Offset(0u),
                Jumpsize(1u)),
            deepiterator::makeIteratorPrescription(
                deepiterator::makeAccessor(),
                deepiterator::makeNavigator(
                    Offset(threadIdx.x),
                    Jumpsize_256())));
    
    auto viewSupercellContainer = deepiterator::makeView(supercellContainer, prescriptionSupercell);
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
        
       
        auto viewParticle = deepiterator::makeView(*it, prescriptionParticle);
        
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
        

        auto viewParticle = deepiterator::makeView(*it, prescriptionParticle);
        
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

