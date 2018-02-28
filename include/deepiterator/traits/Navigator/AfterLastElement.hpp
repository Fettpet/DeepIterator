#pragma once

/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief This trait is used as the element after the last one. The trait has
 * two functions: 
 * 1. test(Container*, Index, ContainerSizeFunction): returns true if the 
 * current element is after the last one, false otherwise
 * 2. set(Container*, Index, ContainerSizeFunction): Set the iterator to the
 * element after the last. 
 * Both function has three arguments:
 * 1. Container*: A pointer to the container, over which you itearte
 * 2. Index: The current position of the iterator
 * 3. ContainerSizeFunction: If the number of elements within the container is
 * needed, this argument can be used. Call ContainerSizeFunction(Container*) to 
 * get the number of elements. This could be an expensiv operation.
 * @tparam TContainer The container over which the iteartor walks.
 * @tparam TIndex The type of the index to get a component out of the container.
 * @tparam TContainerCategory An SFINAE type for categories.
 * @tparam TOffset Type of the offset. This is a template of the function.
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
    typename TContainerCategory>
struct AfterLastElement;



} // namespace navigator

} // namespace traits
    
} // namespace hzdr

