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
struct PreviousElement
{
    HDINLINE
    void
    operator() (TContainer*, TIndex&, TRange&)
    {
        // is not implemented
        assert(true); // Specify the trait
    }
};


} // namespace navigator

} // namespace traits
    
} // namespace hzdr


