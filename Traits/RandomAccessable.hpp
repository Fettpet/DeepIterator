#pragma once 
#include "Traits/ContainerCategory"

namespace hzdr 
{
namespace traits
{
    
template<typename TCategory>
struct RandomAccessable;

template<>
struct RandomAccessable<hzdr::traits::details::ArrayBased>
{
    static const bool value = true;
};

template<>
struct RandomAccessable<hzdr::traits::details::ListBased>
{
    static const bool value = false;
};

} // namespace traits
}// namespace hzdr
