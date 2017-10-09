#pragma once
#include "Traits/ContainerCategory.hpp"
#include "Iterator/Categorie/ArrayLike.hpp"
#include "Iterator/Categorie/DoublyLinkListLike.hpp"

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

template<>
struct IndexType<hzdr::container::categorie::ArrayLike>
{
    typedef int_fast32_t type; 
};

template<>
struct IndexType<hzdr::container::categorie::DoublyLinkListLike>
{
    typedef int_fast32_t type; 
};


template<>
struct IndexType<hzdr::details::UndefinedType>
{
    typedef int type; 
};

} // namespace traits

} // namespace hzdr
