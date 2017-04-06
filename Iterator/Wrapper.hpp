/**
 * @brief While the collectiv iteration over all values in the container, it is
 * possible, that the element is not valid. But after calling ++operator it is
 * valid. This class has two function:
 * operator bool: return true if the value is valid and false otherwise
 * operator* return the value.
 * 
 */
#pragma once
#include "Iterator/Collective.hpp"

namespace hzdr
{

template<typename TElement,
         typename Collectivity>
struct Wrapper
{
    typedef TElement            ElementType;
    typedef ElementType* const  ElementPtr;
    
    Wrapper(ElementPtr ptr):
        ptr(ptr)
    {}
    
    Wrapper(std::nullptr_t):
        ptr(nullptr)
    {}
    
    inline 
    ElementType&
    operator*()
    {
        return *ptr;
    }
    
    inline
    explicit
    operator bool()
    {
        return ptr != nullptr;
    }
    
protected:
    ElementPtr ptr;
    
};




template<typename TElement>
struct Wrapper<TElement, Collectivity::None>
{
    typedef TElement        ElementType;
    typedef ElementType*    ElementPtr;
    
    
    Wrapper(ElementPtr ptr):
        ptr(ptr)
    {}
    
    Wrapper(std::nullptr_t):
        ptr(nullptr)
    {}
    
    inline 
    ElementType&
    operator*()
    {
        return *ptr;
    }
    
    constexpr 
    explicit
    operator bool()
    {
        return true;
    }
    
protected:
    ElementPtr ptr;
    
};


}