#pragma once
#include <cassert>
#include "PIC/Supercell.hpp"

/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief This trait is used as the element before the first one. The trait has
 * two functions: 
 * 1. test(Container*, Index, ContainerSizeFunction): returns true if the 
 * current element is before the first one, false otherwise
 * 2. set(Container*, Index, ContainerSizeFunction): Set the iterator to the
 * element before the first one. 
 * Both function has three arguments:
 * 1. Container*: A pointer to the container, over which you itearte
 * 2. Index: The current position of the iterator
 * 3. ContainerSizeFunction: If the number of elements within the container is
 * needed, this argument can be used. Call ContainerSizeFunction(Container*) to 
 * get the number of elements. This could be an expensiv operation.
 * @tparam TContainer The container over which the iteartor walks.
 * @tparam TIndex The type of the index to get a component out of the container.
 * @tparam TContainerCategory An SFINAE type for categories.
 * @tparam TSizeFunction This is used to give a function, which calculate the 
 * size of the container, to the trait. It is a template of the function, not of
 * the trait.
 */
namespace hzdr
{
namespace traits
{
namespace navigator
{
template<
    typename TContainer,
    typename TIndex,
    typename TOffset,
    typename TContainerCategory>
struct BeforeFirstElement
{
    template<typename TSizeFunction>
    HDINLINE
    void
    set(TContainer*, TIndex&, TOffset&, TSizeFunction&)
    const;

    template<typename TSizeFunction>
    HDINLINE
    bool
    set(TContainer*, TIndex&, TOffset&, TSizeFunction&)
    const;
    
};

}// namespace navigator

} // namespace traits
    
} // namespace hzdr

