#pragma once
/**
 * @author Sebastian Hahn t.hahn < at > hzdr.de
 * @brief Each container category need to specify one range type. This is the 
 * type of the distance between two iterator points
 * 
 */
namespace hzdr 
{
namespace traits
{

template<
    typename TContainer, 
    typename SFIANE = void
>
struct RangeType;



}// namespace traits
}// namespace hzdr


