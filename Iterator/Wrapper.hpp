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
#include "Traits/IsIndexable.hpp"
#include "Traits/NumberElements.hpp"
#include "Traits/NeedRuntimeSize.hpp"

namespace hzdr
{




template<typename TElement,
         typename TCollectivity>
struct Wrapper
{
    typedef TElement            ElementType;
    typedef ElementType* const  ElementPtr;
    
    template<typename TContainer,
            typename TComponent,
            typename TIndex,
            typename TRuntimeVariable>
    HDINLINE
    Wrapper(ElementPtr ptr, 
            TContainer const * const containerPtr, 
            TComponent const * const componentenPtr, 
            const TIndex& pos, 
            const TRuntimeVariable& run,
            typename std::enable_if<traits::IsIndexable<TContainer>::value, int>::type* = 0):
        ptr(ptr)
    {
        const int_fast32_t elem = traits::NeedRuntimeSize<TContainer>::test(containerPtr)? run.getNbElements()  : traits::NumberElements< TContainer>::value;
        result = pos >= 0 and pos < elem;
    }
    
        template<typename TContainer,
            typename TComponent,
            typename TIndex,
            typename TRuntimeVariable>
    HDINLINE
    Wrapper(ElementPtr ptr, 
            TContainer const * const containerPtr, 
            TComponent const * const componentenPtr, 
            const TIndex& pos, 
            const TRuntimeVariable& runtimeVariables,
            typename std::enable_if<not traits::IsIndexable<TContainer>::value, int>::type* = 0):
        ptr(ptr)
    {
        result = ptr != nullptr;
    }
    
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
        return result;
    }
    
protected:
    ElementPtr ptr;
    bool result;
    
};


    




template<typename TElement>
struct Wrapper<TElement,
                Collectivity::None>
{
    typedef TElement        ElementType;
    typedef ElementType*    ElementPtr;
    
    template<typename TContainer,
            typename TComponent,
            typename TIndex,
            typename TRuntimeVariable>
    HDINLINE
    Wrapper(ElementPtr ptr, 
            TContainer const * const containerPtr, 
            TComponent const * const componentenPtr, 
            const TIndex& pos, 
            const TRuntimeVariable& run,
            typename std::enable_if<traits::IsIndexable<TContainer>::value, int>::type* = 0):
        ptr(ptr)
    {
        const int_fast32_t elem = traits::NeedRuntimeSize<TContainer>::test(containerPtr)? run.getNbElements()  : traits::NumberElements< TContainer>::value;
        result = pos >= 0 and pos < elem;
    }
    
        template<typename TContainer,
            typename TComponent,
            typename TIndex,
            typename TRuntimeVariable>
    HDINLINE
    Wrapper(ElementPtr ptr, 
            TContainer const * const containerPtr, 
            TComponent const * const componentenPtr, 
            const TIndex& pos, 
            const TRuntimeVariable& runtimeVariables,
            typename std::enable_if<not traits::IsIndexable<TContainer>::value, int>::type* = 0
           ):
        ptr(ptr),
        result(ptr != nullptr)
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
        return result;
    }
    
protected:
    bool result;
    ElementPtr ptr;
    
};


}
