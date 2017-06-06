/**
 * @author  
 * @brief The runtimetuple is needed to store all variable needed at runtime. 
 * The runtimetuple must have for functions:
 * 1. getOffset() which returns the number of elements, which are overjumped at
 * the beginning of the iteration.
 * 2. getNbElements() returns the number of elements within the datastructure
 * 3. getJumpsize() returns the distance to the next element. 
 */

#pragma once
#include <type_traits>
#include "Definitions/hdinline.hpp"
namespace hzdr 
{
namespace runtime 
{
    struct TupleFull
    {
        HDINLINE
        TupleFull() = default;
        
        HDINLINE
        TupleFull(const TupleFull& other) = default;
        
        HDINLINE
        TupleFull(const int_fast32_t& offset, 
                  const int_fast32_t& nbElements,
                  const int_fast32_t& jumpsize):
            nbElements(nbElements),
            offset(offset), 
            jumpsize(jumpsize)
        {}
        
        HDINLINE
        int_fast32_t getOffset() const
        {
            return offset;
        }
        
        HDINLINE
        int_fast32_t getNbElements() const
        {
            return nbElements;
        }
        
        HDINLINE
        int_fast32_t getJumpsize() const 
        {
            return jumpsize;
        }
        
        int_fast32_t nbElements, offset, jumpsize;
        
    };
    
    
#ifdef _OPENMP
    struct TupleOpenMP
    {
        HDINLINE
        TupleOpenMP() = default;
        
        HDINLINE
        TupleOpenMP(const TupleOpenMP& other) = default;
        
        HDINLINE
        TupleOpenMP( const int_fast32_t& nbElements):
            nbElements(nbElements)
        {}
        
        HDINLINE
        int_fast32_t getOffset() const
        {
            return omp_get_thread_num();
        }
        
        HDINLINE
        int_fast32_t getNbElements() const
        {
            return nbElements;
        }
        
        HDINLINE
        int_fast32_t getJumpsize() const 
        {
            return omp_get_num_threads();
        }
        
        int_fast32_t nbElements;
        
    };
#endif
}; // 
    
}// hzdr
