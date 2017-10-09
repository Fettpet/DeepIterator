/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief Each container categorie need to determine if it is a random accessable.
 * A container categorie is random accessable if there is a way to overjump some
 * values easily.
 */
#pragma once
#include "Iterator/Categorie/ArrayLike.hpp"
#include "Iterator/Categorie/DoublyLinkListLike.hpp"

namespace hzdr 
{
namespace traits
{
template<typename TContainerCategorie>
struct IsBidirectional
{
    static const bool value = false;
};

template<>
struct IsBidirectional<hzdr::container::categorie::ArrayLike>
{
    static const bool value = true;
};

template<>
struct IsBidirectional<hzdr::container::categorie::DoublyLinkListLike>
{
    static const bool value = true;
};

template<>
struct IsBidirectional<details::UndefinedType>
{
    static const bool value = false;
};

}// namespace traits
}// namespace hzdr


