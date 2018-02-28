#pragma once


/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief We use this trait to check whether an iterators position is equal to an
 * other iterators position. This means it1 == it2. The trait need the operator().
 * It has four arguments:
 * 1. A pointer to the container of the first iterator,
 * 2. The index of the first iterator,
 * 3. A pointer to the container of the second iterator 
 * 4. The index of the second iterator
 * 
 * @tparam TContainer The container over which the iteartor walks.
 * @tparam TComponent The component of the container.
 * @tparam TIndex The type of the index to get a component out of the container.
 * @tparam TContainerCategory An SFINAE type for categories.
 * 
 * @return true, if the first iterator is at same position as the second one, false otherwise
 *
 */
namespace hzdr
{
namespace traits
{
namespace accessor
{
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TContainerCategory>
struct Equal;


} // namespace accessor

} // namespace traits
    
} // namespace hzdr
