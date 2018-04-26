/**
 * @author Sebastian Hahn 
 * @brief This forward implementation is needed, since the std forward throughs 
 * warning. The std forward hasnt HDINLINE. This is the diffence between both
 * implementations.
 * 
 */
#pragma once
#include "deepiterator/definitions/hdinline.hpp"

namespace deepiterator 
{
template <class T>
HDINLINE T&& forward(typename std::remove_reference<T>::type& t) noexcept
{
    return static_cast<T&&>(t);
}

template <class T>
HDINLINE T&& forward(typename std::remove_reference<T>::type&& t) noexcept
{
    static_assert(!std::is_lvalue_reference<T>::value,
                  "Can not forward an rvalue as an lvalue.");
    return static_cast<T&&>(t);
}
    
}
