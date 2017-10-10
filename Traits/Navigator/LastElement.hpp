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
struct LastElement
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (TContainer*, TIndex&, TRange&, TRange&, TSizeFunction&)
    {
        // is not implemented. 
        assert(true); // Specify the trait
    }
    
    HDINLINE void UNDEFINED();
};


} // namespace navigator
} // namespace traits
    
} // namespace hzdr


