
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
    typename TContainerCategory>
struct AfterLastElement
{
    template<typename TRangeFunction>
    HDINLINE
    void
    operator() (TContainer*, TIndex const &, TRangeFunction const &)
    const
    {
        // is not implemented. 
        assert(true); // Specify the trait
    }
};



} // namespace navigator

} // namespace traits
    
} // namespace hzdr

