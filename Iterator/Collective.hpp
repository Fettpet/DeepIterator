#ifdef _OPENMP
#include <omp.h>
#endif
#include "Definitions/hdinline.hpp"
/**
 * @brief The Collectiv policy needs two functions: 
 * 1. bool isMover() specifies the worker which moves the current element within
 * the iterator to the next one. 
 * 2. void sync() After the move all worker must be synchronised
 * 
 */

#pragma once
namespace hzdr
{
namespace Collectivity
{
/**
 * @brief The iterator doesn't 
 * */
struct None
{
    HDINLINE
    constexpr 
    bool 
    isMover()
    const
    {
        return true;
    }
    
    HDINLINE 
    void 
    sync()
    {}
    
    
};
#ifdef _OPENMP
struct OpenMPNotIndexable
{
    HDINLINE
    bool 
    isMover()
    {
        return true;
    }
    
    HDINLINE 
    void 
    sync()
    {
#pragma omp barrier 
    }
    
};


struct OpenMPIndexable
{
    constexpr 
    bool 
    isMover()
    const
    {
        return true;
    }
    
    HDINLINE 
    void 
    sync()
    {
//#pragma omp barrier 
    }
    
};
#endif



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
    void 
    allocSharedMem(int**& sharedMem, int* globalMem)
    {
        __shared__ int* arr[1];
        sharedMem=arr;
    }
#else
    HDINLINE
    void
    sync()
    {

    }
    
    HDINLINE
    void 
    allocSharedMem(int**& , int* )
    {
    }

    
#endif
    

    
    HDINLINE
    bool
    isMover(int ID)
    {
        return ID == 0;
    }
};

}// namespace Collectiv
}// namespace hzdr
