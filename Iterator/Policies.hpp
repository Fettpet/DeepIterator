/**
 * 
 * 
 * 
 */



#pragma once
#include "Definitions/hdinline.hpp"
namespace hzdr 
{
/**
 * @brief This policy is used to decide in which direction the iterator walk 
 * through the datastructure.
 * 
 * This is importend for our implementation of the navigator. We had two cases 
 * implemented:
 * Forward: start at the first entry and go to the last one.
 * Backward: start at the last entry and go to the first one
 * 
 * \see Navigator
 */

namespace Direction
{
template<uint_fast32_t jumpsize>
struct Forward 
{
    static_assert(jumpsize != 0, "Jumpsize need to be greater than 0");
    
    constexpr
    uint_fast32_t 
    getJumpsize()
    const
    {
        return jumpsize;
    }
};

template<uint_fast32_t jumpsize>
struct Backward
{
    static_assert(jumpsize != 0, "Jumpsize need to be greater than 0");
    
    constexpr
    uint_fast32_t 
    getJumpsize()
    const
    {
        return jumpsize;
    }
};

    
}

/**
 * @brief The NoChild is used to define the last layer in a nested Iterator
 */
struct NoChild {};
    
}
