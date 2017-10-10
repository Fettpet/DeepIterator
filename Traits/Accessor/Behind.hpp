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
namespace accessor
{
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TContainerCategory>
struct Behind
{
    HDINLINE
    bool
    operator() (TContainer*, TIndex&, TContainer*, TIndex&)
    {
        // is not implemented. Specify the trait
        assert(true); 
    }
};



} // namespace accessor
} // namespace traits
    
} // namespace hzdr

