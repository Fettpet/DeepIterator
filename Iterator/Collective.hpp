#ifdef _OPENMP
#include <omp.h>
#endif
#include "Definitions/hdinline.hpp"
/**
 * \namespace Collectivity
 * @author Sebastian Hahn < t.hahn @ hzdr.de >
 * 
 * @brief The collectivity is used to design the parallel stuff. It need to provide
 * three functions: An offset, a jumpsize and a synchonization function. 
 * 
 * The offset is used to specfiy the first component, within the container. Each thread
 * should have an unique offset. The offset should be smaller than the number of 
 * threads. The jumpsize should be the number of threads.
 * The heads of the collectivity functions are:
 * 1. void sync() After the move all worker must be synchronised
 * 2. uint_fast32_t offset(): returns the id of the thread within a warp
 * 3. uint_fast32_t nbThreads(): number of threads within a warp
 */

#pragma once
namespace hzdr
{
namespace Collectivity
{
/**
 * @brief The None collectivity is used in a sequenciel application. The offset 
 * is zero. The number of threads are 1. The sync function is empty.
 * */
struct None
{

    HDINLINE 
    void 
    sync()
    const
    {}
    
    HDINLINE
    constexpr
    uint_fast32_t
    offset()
    const
    {
        return 0;
    }
    
    HDINLINE
    constexpr
    uint_fast32_t
    nbThreads()
    const
    {
        return 1;
    }
    
};


/**
 * @brief The CudaIndexable is used for a implementation on GPU with Cuda.
 * 
 * The CudaIndexable use only the thread level and only the x dimension. Because
 * of problems with the compiler, you need to specify each function two times. 
 * The first one is within an #ifdef __CUDACC__. These are functions for the GPU.
 * The second implementation is for the CPU.
 */

struct CudaIndexable
{
#ifdef __CUDACC__
    __device__
    void 
    sync()
    {
        __syncthreads();
    }
    
    
    __device__
    uint_fast32_t
    offset()
    const
    {
        return threadIdx.x;
    }
    
    __device__
    uint_fast32_t
    nbThreads()
    const
    {
        return blockDim.x;
    }
#else
    HDINLINE
    void
    sync()
    {

    }

    HDINLINE
    uint_fast32_t
    offset()
    const
    {
        return 0;
    }
    
    HDINLINE
    uint_fast32_t
    nbThreads()
    const
    {
        return 1;
    }
#endif
    

};

}// namespace Collectiv
}// namespace hzdr
