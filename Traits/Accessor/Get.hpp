#pragma once


/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief This trait is used to get a component out of the container. We had two 
 * arguments:
 * 1. A pointer to the container of the iterator,
 * 2. The index of the iterator
 * 
 * @tparam TContainer The container over which the iteartor walks.
 * @tparam TComponent The component of the container.
 * @tparam TIndex The type of the index to get a component out of the container.
 * @tparam TContainerCategory An SFINAE type for categories.
 */
#include "Definitions/hdinline.hpp"
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
struct Get
{
    HDINLINE
    TComponent& 
    operator() (TContainer*, TIndex&);
    
    bool Debugging = false;
} ;



} // namespace accessor

    
} // namespace traits
    
} // namespace hzdr
