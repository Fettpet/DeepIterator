#pragma once
#include <cassert>
#include "PIC/Supercell.hpp"

/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief This trait is used to set the iterator to the first element. If there
 * are not enough elements (e.g. empty container) The iterator is set to the 
 * AfterLastElement or the BeforeFirstElement. The trait need the operator() with
 * three arguments:
 * 1. A pointer to the container
 * 2. A reference to the index
 * 3. An offset. This is the distance between the first element and the first 
 * iterator position.
 * @tparam TContainer The container over which the iteartor walks.
 * @tparam TIndex The type of the index to get a component out of the container.
 * @tparam TContainerCategory An SFINAE type for categories.
 * @tparam TOffset Type of the offset. This is a template of the function, not
 * of the trait.
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
struct FirstElement{
    
    template<typename TOffset>
    HDINLINE
    void
    operator() (
        TContainer*, 
        TIndex&, 
        TOffset const &);
    
};

} // namespace navigator
} // namespace traits
    
} // namespace hzdr
