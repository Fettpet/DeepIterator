/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief This trait is used to decide the indextype. The indextype is used to 
 * specify the position to get the current component out of the container.
 */
#pragma once
#include "PIC/Supercell.hpp"

namespace hzdr
{
namespace details
{
struct UndefinedType;
} // namespace details
namespace traits
{

template<
    typename TContainer, 
    typename SFINAE = void>
struct IndexType
{
    typedef int_fast32_t type; 
};

template<typename TFrame>
struct IndexType<hzdr::Supercell<TFrame> >
{
    typedef TFrame* type; 
};


} // namespace traits

} // namespace hzdr
