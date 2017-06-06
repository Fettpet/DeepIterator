/**
 * @author Sebastian Hahn ( t.hahn@hzdr.de )
 * @brief While the collectiv iteration over all values in the container, it is
 * possible, that the element is not valid. But after calling ++operator it is
 * valid. This class has two function:
 * operator bool: return true if the value is valid and false otherwise
 * operator* return the value.
 * 
 */
#pragma once
#include "Iterator/Collective.hpp"
#include "Definitions/hdinline.hpp"

namespace hzdr
{

template<typename TElement,
         typename Collectivity>
struct Wrapper
{
    typedef TElement            ElementType;
    typedef ElementType* const  ElementPtr;
    
    HDINLINE
    Wrapper(ElementPtr ptr):
        ptr(ptr)
    {}
    
    HDINLINE
    Wrapper(std::nullptr_t):
        ptr(nullptr)
    {}
    
    HDINLINE
    ElementType&
    operator*()
    {
        return *ptr;
    }
    
    HDINLINE
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
    
    HDINLINE
    Wrapper(ElementPtr ptr):
        ptr(ptr)
    {}
    
    HDINLINE
    Wrapper(std::nullptr_t):
        ptr(nullptr)
    {}
    
    HDINLINE
    ElementType&
    operator*()
    {
        return *ptr;
    }
    
    HDINLINE
    explicit
    operator bool()
    {
        return ptr != nullptr;
    }
    
protected:
    ElementPtr ptr;
    
};


}