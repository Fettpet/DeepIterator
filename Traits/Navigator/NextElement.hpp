#pragma once
#include <cassert>
#include "PIC/Supercell.hpp"

/**
 * @author Sebastian Hahn t.hahn <at> hzdr.de
 *
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
    typename TRange,
    typename TContainerCategory>
struct NextElement
{
    template<typename TContainerSize>
    HDINLINE
    TRange
    operator() (TContainer*, TIndex&, TRange const &, TContainerSize&)
    {
        static_assert(true, "You need to specify the NextElement trait");
    }
};


} // namespace navigator

} // namespace traits
    
} // namespace hzdr

