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
 * Forward: start at the first entry and go to the last one.
 * Backward: start at the last entry and go to the first one
 */
enum class Direction {Forward, Backward};



struct NoChild {};
    
}