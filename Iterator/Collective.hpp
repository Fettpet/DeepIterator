#ifdef _OPENMP
#include <omp.h>
#endif
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
    constexpr 
    bool 
    isMover()
    {
        return true;
    }
    
    inline 
    void 
    sync()
    {}
    
    
};
#ifdef _OPENMP
struct OpenMPNotIndexable
{
    inline
    bool 
    isMover()
    {
        return true;
    }
    
    inline 
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
    {
        return true;
    }
    
    inline 
    void 
    sync()
    {
//#pragma omp barrier 
    }
    
};
#endif
}
}
