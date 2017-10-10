#pragma once
#include "Traits/ContainerCategory.hpp"
#include "PIC/Supercell.hpp"

namespace hzdr
{
namespace details
{
struct UndefinedType;
} // namespace details
namespace traits
{

template<typename TContainer>
struct IndexType
{
    typedef int_fast32_t type; 
};

template<typename TFrame>
struct IndexType<hzdr::Supercell<TFrame> >
{
    typedef TFrame* type; 
};


template<>
struct IndexType<hzdr::details::UndefinedType>
{
    typedef int type; 
};

} // namespace traits

} // namespace hzdr
