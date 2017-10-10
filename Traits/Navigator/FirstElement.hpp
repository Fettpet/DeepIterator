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
    typename TOffset,
    typename TContainerCategory>
struct FirstElement{};

} // namespace navigator
} // namespace traits
    
} // namespace hzdr
