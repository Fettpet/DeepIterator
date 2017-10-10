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
namespace details
{
    
}
    
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TContainerCategory>
struct Ahead
{
    HDINLINE
    bool
    operator() (TContainer*, TIndex&, TContainer*, TIndex&)
    {
        // is not implemented. Specify the trait
        assert(true); 
    }
    
    // this function is used to 
    HDINLINE void UNDEFINED(){};
};




} // namespace accessor
} // namespace traits
    
} // namespace hzdr
