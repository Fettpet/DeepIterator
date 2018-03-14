/* Copyright 2018 Sebastian Hahn

 * This file is part of DeepIterator.
 *
 * DeepIterator is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DeepIterator is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once
#include "deepiterator/definitions/forward.hpp" 
#include <type_traits>
#include "deepiterator/traits/Traits.hpp"

#include <cassert>

namespace hzdr 
{

/**
* \struct Navigator
@brief This is the default implementation of the navigator. The task of the 
navigator is to define the first element, the next element and an after last 
element. If the navigator is bidirectional it need also a last element, a 
previous element and a before first element. 

The navigator has two traits for parallel 
walking through the container. The first one is T_Offset. This is used to get the 
distance from the first element of the container to the first element which will
be accessed. This trait can be used to map the thread ID (for example offset 
= threadIdx.x). The second trait is the jumpsize. The jumpsize is the distance
between two iterator elements. The number of threads can be mapped on the jump-
size. With this two traits you can go parallel over all elements and touch each
element only one times. 
 
We had three/six traits for the behaviour of the container. The first three
traits are :
1. define the first element of the container,
2. define a next element of the container,
3. define a after last element of the container.
If the navigator is bidirectional three additional traits are needed
4. define the last element within the container
5. define a previous element of the container
6. define a before first element of the container.
The navigator use this 8 traits to define methodes for parallel iteration though
the container.
 
@tparam T_Container Type of the container,
@tparam T_Component Type of the component of the container.
@tparam T_Offset Policy to get the offset. You need to specify the () operator.
@tparam T_Jumpsize Policy to specify the Jumpsize. It need the operator ().
@tparam T_Index Type of the index. The index is used to specify the iterator 
position.
@tparam T_ContainerSize Trait to specify the size of a container. It need the 
function operator()(T_Container*). T_Container is a pointer to the container 
instance over which the iterator walks.
@tparam T_FirstElement Trait to set the index to the first element. It need the 
function operator()(T_Container*, T_Index&, const T_Range). T_Range is the result 
type of T_Offset's (). T_Container is a pointer to the container 
instance over which the iterator walks. T_Index is used to describe the position.
T_Range is the offset.
@tparam T_NextElement Trait to set the index to the next element. The trait need 
the function T_Range operator()(T_Container*, T_Index&, T_Range). The T_Range 
parameter is used to handle the jumpsize. The result of this function is the 
remaining jumpsize. A little example. Your container has 10 elements and your
iterator is the the 8 element. Your jumpsize is 5. This means the new position
would be 13. So the result of the function is 3, the remaining jumpsize.
@tparam T_AfterLastElement This Trait is used to check whether the iteration is 
after the last element. The function header is 
bool operator()(T_Container*, T_Index&). It returns true, if the end is reached, 
and false otherwise.
@tparam T_LastElement This trait gives the last element which the iterator would
access, befor the end is reached, in a forward iteration case. The function 
head is operator()(T_Container*, T_Index&, const T_Range). This trait is only 
needed if the navigator is bidirectional. 
@tparam T_PreviousElement Trait to set the index to the previous element. The 
trait need the function T_Range operator()(T_Container*, T_Index&, T_Range). This 
trait is only needed if the navigator is bidirectional. For fourther 
informations see T_NextElement.
@tparam T_BeforeFirstElement Used to check whether the iterator is before the
first element. The function header is bool operator()(T_Container*, T_Index&). 
It returns true, if the end is reached, and false otherwise.
@tparam isBidirectional Set the navigator to bidirectional (true) or to forward
only (false)
*/
template<
    typename T_Container,
    typename T_Component,
    typename T_Offset,
    typename T_Jumpsize,
    typename T_Index,
    typename T_ContainerSize,
    typename T_Range,
    typename T_FirstElement,
    typename T_NextElement,
    typename T_AfterLastElement,
    typename T_LastElement = hzdr::details::UndefinedType,
    typename T_PreviousElement = hzdr::details::UndefinedType,
    typename T_BeforeFirstElement = hzdr::details::UndefinedType,
    bool isBidirectional = false
>
struct Navigator
{
// define the types 
    using ContainerType = typename std::decay<T_Container>::type;
    using ContainerPtr = ContainerType*;
    using ContainerRef = ContainerType&;
    using ComponentType = T_Component;
    using ComponentPtr = ComponentType*;
    using JumpsizeType = T_Jumpsize;
    using OffsetType = T_Offset;
    using IndexType = T_Index;
    using RangeType = T_Range;
    using NumberElements = T_ContainerSize;
    using FirstElement = T_FirstElement;
    using NextElement = T_NextElement;
    using AfterLastElement = T_AfterLastElement;
    using LastElement = T_LastElement;
    using PreviousElement = T_PreviousElement;
    using BeforeFirstElement = T_BeforeFirstElement;

    
public:
// the default constructors
    HDINLINE Navigator() = default;
    HDINLINE Navigator(Navigator const &) = default;
    HDINLINE Navigator(Navigator &&) = default;
    HDINLINE ~Navigator() = default;
    HDINLINE Navigator& operator=(const Navigator&) = default;
    HDINLINE Navigator& operator=(Navigator&&) = default;

    
    /**
     * @brief Set the offset and the jumpsize to the given values
       @param offset the distance from the start to the first element
       @param jumpsize distance between two elements
    */
    HDINLINE
    Navigator(
        OffsetType && offset,
        JumpsizeType && jumpsize
    ):
        offset(hzdr::forward<OffsetType>(offset)),
        jumpsize(hzdr::forward<JumpsizeType>(jumpsize)),
        containerSize(),
        firstElement(),
        nextElement(),
        afterLastElement(),
        lastElement(),
        previousElement(),
        beforeFirstElement()
    {}
    
    
    /**
     * @brief The function moves the iterator forward to the next element. 
     * @param index in: current position of iterator; out: position of the 
     * iterator after the move.
     * @param containerPtr pointer to the container, over which we iterate
     * @param distance number of elements you like to overjump.
     * @result the distance from the end element to the hypothetical position
     * given by the distance parameter
     */
    HDINLINE
    auto
    next(
        ContainerPtr containerPtr,  
        IndexType & index,
        RangeType const & distance
    )
    -> RangeType
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        // We jump over distance * jumpsize elements
        auto remainingJumpsize = nextElement(
            containerPtr, 
            index,  
            static_cast<RangeType>(jumpsize()) * distance,
            containerSize
        );
        
        // we need the distance from the last element to the current index 
        // position this is a round up
        return (remainingJumpsize + jumpsize() - static_cast<RangeType>(1)) /
               jumpsize();
    }
    
    
    /**
     * @brief The function moves the iterator backward to the previous element. 
     * This function is only enabled, if the navigator is bidirectional.
     * @param containerPtr pointer to the container, over which we iterate
     * @param index in: current position of iterator; out: position of the 
     * iterator after the move.
     * @param distance number of elements you like to overjump.
     * @result the distance from the end element to the hypothetical position
     * given by the distance parameter
     */
    template< bool T=isBidirectional>
    HDINLINE
    auto
    previous(
        ContainerPtr containerPtr,  
        IndexType & index,
        RangeType distance
    )
    -> typename std::enable_if<
        T==true,
        RangeType
    >::type
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        // We jump over distance * jumpsize elements
        auto && off = offset();
        auto && jump = jumpsize();
        auto remainingJumpsize = previousElement(
            containerPtr, 
            index,
            static_cast<RangeType>(jump) * distance,
            containerSize
        );
        if(remainingJumpsize == 0)
        {
            auto indexCopy = index;
            remainingJumpsize = previousElement(
                containerPtr, 
                indexCopy,
                off,
                containerSize
            );
            return (remainingJumpsize + jump - static_cast<RangeType>(1)) 
             / jump;
        }
        else 
        {
            return (remainingJumpsize + jump - static_cast<RangeType>(1) + off) 
             / jump ;
    
        }

        // we need the distance from the last element to the current index 
        // position
    }
    
    
    /**
     * @brief set the iterator to the first element
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: first element of the iterator.
     */
    HDINLINE 
    auto
    begin(
        ContainerPtr containerPtr,  
        IndexType & index
    )
    -> void
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        firstElement(
            containerPtr,
            index
        );
        nextElement(
            containerPtr, 
            index,  
            static_cast<RangeType>(offset()),
            containerSize
        );
    }
    
    
    /**
     * @brief set the iterator to the last element. 
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: last element of the iterator.
     */
    template< bool T=isBidirectional>
    HDINLINE 
    auto
    rbegin(
        ContainerPtr containerPtr,  
        IndexType & index
    )
    -> void
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        auto nbElementsVar = nbElements(containerPtr);
        // -1 since we dont like to jump outside
        auto nbJumps = (nbElementsVar - offset() - static_cast<RangeType>(1)) 
                      / jumpsize();
        auto lastPosition = nbJumps * jumpsize() + offset();
        // -1 since we need the last position
        auto neededJumps = (nbElementsVar - static_cast<RangeType>(1))
                         - lastPosition;

        lastElement(
            containerPtr,
            index,
            containerSize
        );
        if(not isBeforeFirst(
            containerPtr,
            index
        ))
        {
            previousElement(
                containerPtr, 
                index,
                neededJumps,
                containerSize
            );
        }
        
    }
    
    
    /**
     * @brief set the iterator to the after last element
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: index of the after last element
     */
    HDINLINE 
    auto
    end(
        ContainerPtr containerPtr,  
        IndexType & index
    )
    -> void
    {
        afterLastElement.set(
            containerPtr,
            index,
            containerSize
        );
    }
    
    
    /**
     * @brief set the iterator to the last element. It is possible that two 
     * iterators, the first start with begin, the second with last, never meet.
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: index of the before first element
     */
    template< bool T=isBidirectional>
    HDINLINE
    auto
    rend(
        ContainerPtr containerPtr,  
        IndexType & index
    )
    ->     typename std::enable_if<T==true>::type
    {
        beforeFirstElement.set(
            containerPtr,
            index,
            containerSize
        );
    }
    
    
    /**
     * @brief check wheter the index is after the last element
     * @param containerPtr pointer to the container, over which we iterate
     * @param index in: current index position
     * @return true, if the index is after the last element; false otherwise
     */
    HDINLINE 
    auto
    isAfterLast(
        ContainerPtr containerPtr,  
        IndexType const & index
    )
    const
    -> bool
    {
        return afterLastElement.test(
            containerPtr,
            index,
            containerSize
        );
    }
    
    
    /**
     * @brief check wheter the index is before the first element
     * @param containerPtr pointer to the container, over which we iterate
     * @param index in: current index position
     * @return true, if the index is before the first element; false otherwise
     */
    template< bool T=isBidirectional>
    HDINLINE 
    auto
    isBeforeFirst(
        ContainerPtr containerPtr,   
        IndexType const & index
    )
    const
    -> typename std::enable_if<
        T==true,
        bool
    >::type
    {
        IndexType indexCopy(index);
        PreviousElement prev(previousElement);
        return beforeFirstElement.test(
            containerPtr, 
            index, 
            containerSize
        ) || (
        prev(
                containerPtr, 
                indexCopy,
                offset(),
                containerSize
        ) != static_cast<RangeType>(0));
    }
    
    
    /**
     * @brief this function determine the number of elements within the 
     * container
     * @param containerPtr pointer to the container, you like to know the number
     * of elements
     * @return number of elements within the container
     */
    HDINLINE
    auto
    nbElements(ContainerPtr containerPtr)
    const
    -> uint_fast32_t
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        return containerSize(containerPtr);
    }
    
    
    /**
     * @brief this function determine the number of elements over which the
     * navigator goes. I.e sizeContainer / jumpsize
     * @param containerPtr pointer to the container, you like to know the number
     * of elements
     * @return number of elements the navigator can access
     */
    HDINLINE
    auto
    size(ContainerPtr containerPtr)
    const
    -> uint_fast32_t
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        auto const nbElem = nbElements(containerPtr);
        auto const off = offset();
     //   assert(nbElem >= off);
        const int elems = (nbElem - off + jumpsize() - static_cast<RangeType>(1)) 
            / jumpsize();
        return (elems > 0) * elems;
    }
    
//variables
protected:
    OffsetType offset;
    JumpsizeType jumpsize;
    NumberElements containerSize;
    FirstElement firstElement;
    NextElement nextElement;
    AfterLastElement afterLastElement;
    LastElement lastElement;
    PreviousElement previousElement;
    BeforeFirstElement beforeFirstElement;
} ;


/**
 * @brief This navigator is a concept. It has an offset and a jumpsize.
 */

template<
    typename T_Offset,
    typename T_Jumpsize
>
struct Navigator<
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    T_Offset,
    T_Jumpsize,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    false>
{
    using ContainerType = hzdr::details::UndefinedType;
    using ContainerPtr = ContainerType*;
    using ContainerRef = ContainerType&;
    using ComponentType = hzdr::details::UndefinedType ;
    using ComponentPtr = ComponentType*;
    using JumpsizeType = T_Jumpsize;
    using OffsetType = T_Offset;
    using IndexType = hzdr::details::UndefinedType ;
    using RangeType = hzdr::details::UndefinedType ;
    using NumberElements = hzdr::details::UndefinedType ;
    using FirstElement = hzdr::details::UndefinedType ;
    using NextElement = hzdr::details::UndefinedType ;
    using AfterLastElement = hzdr::details::UndefinedType ;
    using LastElement = hzdr::details::UndefinedType ;
    using PreviousElement = hzdr::details::UndefinedType ;
    using BeforeFirstElement = hzdr::details::UndefinedType ;
    // the default constructors
    HDINLINE Navigator() = default;
    HDINLINE Navigator(Navigator const &) = default;
    HDINLINE Navigator(Navigator &&) = default;
    HDINLINE ~Navigator() = default;
    
    /**
     * @brief Set the offset and the jumpsize to the given values
       @param offset the distance from the start to the first element
       @param jumpsize distance between two elements
    */
    template<
        typename T_Offset_,
        typename T_Jumpsize_
    >
    HDINLINE
    Navigator(
            T_Offset_ && offset, 
            T_Jumpsize_ && jumpsize
    ):
        offset(hzdr::forward<T_Offset_>(offset)),
        jumpsize(hzdr::forward<T_Jumpsize_>(jumpsize))
    {}
    
    OffsetType offset;
    JumpsizeType jumpsize;
} ;


/**
 * @brief creates an navigator concept. It needs an offset and the jumpsize
 * @param offset distance from the begining of the container to the first 
 * position of the iterator 
 * @param jumpsize distance between two elements within the container
 * 
 */
template<
    typename T_Offset,
    typename T_Jumpsize
>
HDINLINE
auto 
makeNavigator(
    T_Offset && offset,
    T_Jumpsize && jumpsize
)
-> 
    hzdr::Navigator<
        details::UndefinedType,
        details::UndefinedType,
        typename std::decay<T_Offset>::type,
        typename std::decay<T_Jumpsize>::type,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        false>
{
    using OffsetType = typename std::decay<T_Offset>::type;
    using JumpsizeType = typename std::decay<T_Jumpsize>::type;
    typedef hzdr::Navigator<
        details::UndefinedType,
        details::UndefinedType,
        OffsetType,
        JumpsizeType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        false> ResultType;
    return ResultType(
        hzdr::forward<T_Offset>(offset),
        hzdr::forward<T_Jumpsize>(jumpsize));
}


namespace details
{

    template<
        typename T,
        typename _T= typename std::decay<T>::type,
        typename TContainerType = typename _T::ContainerType,
        typename TOffsetType = typename _T::OffsetType,
        typename TJumpsizeType = typename _T::JumpsizeType,
        typename TIndexType = typename _T::IndexType,
        typename TRangeType= typename _T::RangeType,
        typename TNumberElements= typename _T::NumberElements,
        typename TFirstElement= typename _T::FirstElement,
        typename TNextElement= typename _T::NextElement,
        typename TAfterLastElement= typename _T::AfterLastElement,
        typename TLastElement= typename _T::LastElement,
        typename TPreviousElement= typename _T::PreviousElement,
        typename TBeforeFirstElement= typename _T::BeforeFirstElement
    >
    struct NavigatorTemplates
    {
        using ContainerType = TContainerType;
        using OffsetType = TOffsetType;
        using JumpsizeType = TJumpsizeType;
        using IndexType = TIndexType ;
        using RangeType = TRangeType;
        using NumberElements = TNumberElements;
        using FirstElement = TFirstElement;
        using NextElement = TNextElement;
        using AfterLastElement = TAfterLastElement;
        using LastElement = TLastElement;
        using PreviousElement = TPreviousElement;
        using BeforeFirstElement = TBeforeFirstElement;
    };


template<
    typename T_Container,
    typename T_ContainerNoRef = typename std::decay<T_Container>::type,
    typename T_Navigator,
    typename T_Offset = typename details::NavigatorTemplates<
        T_Navigator
    >::OffsetType,
    typename T_Jumpsize = typename details::NavigatorTemplates<
        T_Navigator
    >::JumpsizeType,
    typename T_Component = typename hzdr::traits::ComponentType<
        T_ContainerNoRef
    >::type,
    typename T_ContainerCategorie = typename hzdr::traits::ContainerCategory<
        T_ContainerNoRef
    >::type,
    typename T_ContainerSize = typename hzdr::traits::NumberElements<
        T_ContainerNoRef
    >,
    typename T_Index = typename hzdr::traits::IndexType<
        T_ContainerNoRef,
        T_ContainerCategorie
    >::type,
    typename T_Range = typename hzdr::traits::RangeType<
        T_ContainerNoRef,
        T_ContainerCategorie
    >::type,
    typename T_FirstElement = typename hzdr::traits::navigator::FirstElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_AfterLastElement = typename hzdr::traits::navigator::AfterLastElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_NextElement = typename hzdr::traits::navigator::NextElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    typename T_LastElement = typename hzdr::traits::navigator::LastElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie>,
    typename T_PreviousElement = typename hzdr::traits::navigator::PreviousElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    typename T_BeforeFirstElement = typename hzdr::traits::navigator::BeforeFirstElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    bool isBidirectional = not std::is_same<
        T_LastElement, 
        hzdr::details::UndefinedType
    >::value,
     typename = typename std::enable_if< 
        std::is_same<
            hzdr::Navigator<
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                T_Offset,
                T_Jumpsize,
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                hzdr::details::UndefinedType,
                false
            >,
            typename std::decay<T_Navigator>::type
        >::value
    >::type
    
>
auto
HDINLINE
makeNavigator(T_Navigator && navi)
-> 
hzdr::Navigator<
    T_ContainerNoRef,
    T_Component,
    T_Offset,
    T_Jumpsize,
    T_Index,
    T_ContainerSize,
    T_Range,
    T_FirstElement,
    T_NextElement,
    T_AfterLastElement,
    T_LastElement,
    T_PreviousElement,
    T_BeforeFirstElement,
    isBidirectional
>

{ 
    using ResultType = hzdr::Navigator<
        T_ContainerNoRef,
        T_Component,
        T_Offset,
        T_Jumpsize,
        T_Index,
        T_ContainerSize,
        T_Range,
        T_FirstElement,
        T_NextElement,
        T_AfterLastElement,
        T_LastElement,
        T_PreviousElement,
        T_BeforeFirstElement,
        isBidirectional
    > ;
        

    auto && result = ResultType(
        hzdr::forward<T_Offset>(navi.offset),
        hzdr::forward<T_Jumpsize>(navi.jumpsize)
    );

    return result;
}


} // namespace details


/**
 * @brief creates an iterator
 * @tparam container type of the container
 * @param offset distance from the start of the container to the first element 
 * of the iterator
 * @param jumpsize distance between to elements within the container
 */

template<
    typename T_Container,
    typename T_ContainerNoRef = typename std::decay<T_Container>::type,
    typename T_Offset,
    typename T_Jumpsize,
    typename T_Component = typename hzdr::traits::ComponentType<
        T_ContainerNoRef
    >::type,
    typename T_ContainerCategorie = typename hzdr::traits::ContainerCategory<
        T_ContainerNoRef
    >::type ,
    
    typename T_ContainerSize = typename hzdr::traits::NumberElements<
        T_ContainerNoRef
    >::type,
    typename T_Index = typename hzdr::traits::IndexType<
        T_ContainerNoRef,
        T_ContainerCategorie
    >::type,
    typename T_Range = typename hzdr::traits::RangeType<
        T_ContainerNoRef,
        T_ContainerCategorie
    >::type,
    typename T_FirstElement = typename hzdr::traits::navigator::FirstElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >::type,
    typename T_AfterLastElement = typename hzdr::traits::navigator::AfterLastElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >::type,
    typename T_NextElement = typename hzdr::traits::navigator::NextElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >::type,
    typename T_LastElement = typename hzdr::traits::navigator::LastElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >::type,
    typename T_PreviousElement = typename hzdr::traits::navigator::PreviousElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >::type,
    typename T_BeforeFirstElement = typename hzdr::traits::navigator::BeforeFirstElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >::type,
    bool isBidirectional = not std::is_same<
        T_LastElement, 
        hzdr::details::UndefinedType
    >::value
    
>
auto 
HDINLINE
makeNavigator(
    T_Offset && offset,
    T_Jumpsize && jumpsize
)
-> 
    hzdr::Navigator<
        T_ContainerNoRef,
        T_Component,
        T_Offset,
        T_Jumpsize,
        T_Index,
        T_ContainerSize,
        T_Range,
        T_FirstElement,
        T_NextElement,
        T_AfterLastElement,
        T_LastElement,
        T_PreviousElement,
        T_BeforeFirstElement,
        isBidirectional
    >

{ 
    using ResultType =  hzdr::Navigator<
        T_ContainerNoRef,
        T_Component,
        T_Offset,
        T_Jumpsize,
        T_Index,
        T_ContainerSize,
        T_Range,
        T_FirstElement,
        T_NextElement,
        T_AfterLastElement,
        T_LastElement,
        T_PreviousElement,
        T_BeforeFirstElement,
        isBidirectional
    > ;
    auto && result = ResultType(
        hzdr::forward<T_Offset>(offset),
        hzdr::forward<T_Jumpsize>(jumpsize)
    );
    
    return result;
    
}

}// namespace hzdr
