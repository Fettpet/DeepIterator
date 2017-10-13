/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief Each container categorie need to determine if it is a random accessable.
 * A container categorie is random accessable if there is a way to overjump some
 * values easily. Mostly a container is random accessable if the operator [] is
 * overloaded
 */
#pragma once
#include "Iterator/Categorie/ArrayLike.hpp"
#include "Iterator/Categorie/DoublyLinkListLike.hpp"

namespace hzdr 
{
namespace traits
{
template<typename TContainerCategorie>
struct IsRandomAccessable
{
    static const bool value = false;    
};


template<>
struct IsRandomAccessable<details::UndefinedType>
{
    static const bool value = false;
};


template<>
struct IsRandomAccessable<hzdr::container::categorie::ArrayLike>
{
    static const bool value = true;
};

template<>
struct IsRandomAccessable<hzdr::container::categorie::DoublyLinkListLike>
{
    static const bool value = true;
};

}// namespace traits
}// namespace hzdr
