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
        
        TupleFull(const uint_fast32_t& jumpsize, 
                         const uint_fast32_t& nbElements,
                         const uint_fast32_t& offset):
            nbElements(nbElements),
            offset(offset), 
            jumpsize(jumpsize)
        {}
        
        uint_fast32_t getOffset() const
        {
            return offset;
        }
        
        uint_fast32_t getNbElements() const
        {
            return nbElements;
        }
        
        uint_fast32_t getJumpsize() const 
        {
            return jumpsize;
        }
        
        uint_fast32_t nbElements, offset, jumpsize;
        
    };
    
}; // 
    
}// hzdr