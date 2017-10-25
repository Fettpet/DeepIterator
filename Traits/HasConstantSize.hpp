/**
 * \struct HasConstantSize
 * @author Sebastian Hahn (t.hahn < at > hzdr) 
 * @brief This trait decide whether a container has a constant size, or not. A 
 * size of a container is constant, if number of elements within the container
 * doesn't change while runtime.
 * 
 */

#pragma once


namespace hzdr 
{
namespace traits
{

    
template<typename T>
struct HasConstantSize
{
    const static bool value = false;
};

}// namespace traits
    
}// namespace hzdr

