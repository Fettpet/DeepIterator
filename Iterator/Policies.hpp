/**
 * 
 * 
 * 
 */



#pragma once
namespace Data 
{
/**
 * @brief This policy is used to decide in which direction the Iterator walk 
 * through the datastructure.
 * Forward: start at the first entry and go to the last one.
 * Backward: start at the last entry and go to the first one
 */
enum class Direction {Forward, Backward };

/**
 * @brief This policy is used to decide which is the datastruce. We doesn't use
 * the direct datastruce because they are templates.
 */
enum class Datastructure {Cell, Frame, Supercell, Particle};
    
}