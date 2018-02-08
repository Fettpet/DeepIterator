/**
 * \struct NumberElements
 * @author Sebastian Hahn t.hahn@hzdr.de
 * @brief This is a helper class to get the number of elements within
 * a container. This helper class has one function, size(const containertype&), 
 * which determine the size of the container. If it is not possible, for example
 * an linked list, it return std::limits::<uint>::min()
 */

#pragma once

namespace hzdr 
{
namespace traits
{
template<typename T>
struct NumberElements;

    
} // namespace traits

}// namespace hzdr
