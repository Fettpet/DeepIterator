#pragma once
#include <cassert>
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
    typename TIndex>
struct Equal
{
    HDINLINE
    bool
    operator() (TContainer*, TIndex&, TContainer*, TIndex&)
    {
        // is not implemented. Specify the trait
        assert(true); 
    }
};
}
} // namespace traits
    
} // namespace hzdr
