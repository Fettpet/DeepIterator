/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief Each container categorie need to determine if it is a random accessable.
 * A container categorie is random accessable if there is a way to overjump some
 * values easily. Mostly a container is random accessable if the operator [] is
 * overloaded. 
 */
#pragma once

namespace hzdr 
{
namespace traits
{
template<typename TContainerCategorie, typename SFIANE = void>
struct IsRandomAccessable;


}// namespace traits
}// namespace hzdr

