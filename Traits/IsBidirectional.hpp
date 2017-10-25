/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief A container is bidirectional it is possible to go to the previous element.
 * The deepiterator has the functions --it and it-- if it is bidirectional.
 */
#pragma once

namespace hzdr 
{
namespace traits
{
template<typename TContainerCategorie, typename SFIANE = void>
struct IsBidirectional;

}// namespace traits
}// namespace hzdr


