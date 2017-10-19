/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief A container is bidirectional it is possible to go to the previous element.
 * The deepiterator has the functions --it and it-- if it is bidirectional.
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

}// namespace traits
}// namespace hzdr


