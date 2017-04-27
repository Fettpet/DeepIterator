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

namespace hzdr 
{
namespace runtime 
{
    struct TupleFull
    {
        TupleFull() = default;
        
        TupleFull(const TupleFull& other) = default;
        
        TupleFull(const int_fast32_t& jumpsize, 
                  const int_fast32_t& nbElements,
                  const int_fast32_t& offset):
            nbElements(nbElements),
            offset(offset), 
            jumpsize(jumpsize)
        {}
        
        int_fast32_t getOffset() const
        {
            return offset;
        }
        
        int_fast32_t getNbElements() const
        {
            return nbElements;
        }
        
        int_fast32_t getJumpsize() const 
        {
            return jumpsize;
        }
        
        int_fast32_t nbElements, offset, jumpsize;
        
    };
    
    
#ifdef _OPENMP
    struct TupleOpenMP
    {
        TupleOpenMP() = default;
        
        TupleOpenMP(const TupleOpenMP& other) = default;
        
        TupleOpenMP( const int_fast32_t& nbElements):
            nbElements(nbElements)
        {}
        
        int_fast32_t getOffset() const
        {
            return omp_get_thread_num();
        }
        
        int_fast32_t getNbElements() const
        {
            return nbElements;
        }
        
        int_fast32_t getJumpsize() const 
        {
            return omp_get_num_threads();
        }
        
        int_fast32_t nbElements;
        
    };
#endif
}; // 
    
}// hzdr