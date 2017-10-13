#pragma once

/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief This trait is used to set the index to the previous element. The trait need
 * the operator() with four arguments:
 * 1. A pointer to the container
 * 2. A reference to the index
 * 3. Jumpsize: Distance between the current and previous element.
 * 4. TContainerSize: Trait to get the size of the container. 
 * 
 * @result If the jumpsize is greater than the remaining elements in the container,
 * the result is the "unjumped" elements. i.e  -1 * Hypotetical positon. 
 * 
 * @tparam TContainer The container over which the iteartor walks.
 * @tparam TIndex The type of the index to get a component out of the container.
 * @tparam TContainerCategory An SFINAE type for categories.
 * @tparam TJumpsize Type of the offset. This is a template of the function, not
 * of the trait.
 * @tparam TSizeFunction Function to get the size of a container. You can use 
 * TSizeFunction(TContainer*) to get the number of the container. This is a 
 * template of the function.
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
struct PreviousElement
{
    template<typename TContainerSize, typename TJumpsize>
    HDINLINE
    TRange
    operator() (TContainer*, TIndex&, TJumpsize const &, TContainerSize&);
}


} // namespace navigator

} // namespace traits
    
} // namespace hzdr


