#pragma once
/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief Each container category need to specify one range type. This is the 
 * type of the distance between two iterator points
 * 
 */
#include "Iterator/Categorie/ArrayLike.hpp"
#include "Iterator/Categorie/DoublyLinkListLike.hpp"

namespace hzdr 
{
namespace traits
{
template<typename TContainerCategorie>
struct RangeType;

template<>
struct RangeType<hzdr::container::categorie::ArrayLike>
{
    typedef int_fast32_t type;
};

template<>
struct RangeType<hzdr::container::categorie::DoublyLinkListLike>
{
    typedef int_fast32_t type;
};

template<>
struct RangeType<details::UndefinedType>
{
    typedef void type;
};

}// namespace traits
}// namespace hzdr

