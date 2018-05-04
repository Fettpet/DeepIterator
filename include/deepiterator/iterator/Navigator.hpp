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

namespace deepiterator 
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
@tparam T_BeginElement Trait to set the index to the first element. It need the 
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
@tparam T_EndElement This Trait is used to check whether the iteration is 
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
@tparam T_REndElement Used to check whether the iterator is before the
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
    typename T_Size,
    typename T_Range,
    typename T_BeginElement,
    typename T_NextElement,
    typename T_EndElement,
    typename T_LastElement = deepiterator::details::UndefinedType,
    typename T_PreviousElement = deepiterator::details::UndefinedType,
    typename T_REndElement = deepiterator::details::UndefinedType,
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
    using Size = T_Size;
    using BeginElement = T_BeginElement;
    using NextElement = T_NextElement;
    using EndElement = T_EndElement;
    using LastElement = T_LastElement;
    using PreviousElement = T_PreviousElement;
    using REndElement = T_REndElement;

    
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
        offset(deepiterator::forward<OffsetType>(offset)),
        jumpsize(deepiterator::forward<JumpsizeType>(jumpsize)),
        containerSize(),
        beginElement(),
        nextElement(),
        endElement(),
        lastElement(),
        previousElement(),
        beforeBeginElement()
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
        RangeType remainingJumpsize = nextElement(
            containerPtr, 
            index,  
            static_cast<RangeType>(jumpsize()) * distance,
            containerSize
        );
        
        // we need the distance from the last element to the current index 
        // position this is a round up
        return (remainingJumpsize + 
                static_cast<RangeType>(jumpsize()) -
                static_cast<RangeType>(1)) /
                static_cast<RangeType>(jumpsize()) ;
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
        auto remainingJumpsize = previousElement(
            containerPtr, 
            index,
            static_cast<RangeType>(jumpsize())  * distance,
            containerSize
        );
        if(remainingJumpsize == 0)
        {
            auto indexCopy = index;
            remainingJumpsize = previousElement(
                containerPtr, 
                indexCopy,
                static_cast<RangeType>(offset()) ,
                containerSize
            );

            return (
                remainingJumpsize + 
                static_cast<RangeType>(jumpsize()) -
                static_cast<RangeType>(1)) /
                static_cast<RangeType>(jumpsize());
        }
        else 
        {
            return 
                (remainingJumpsize + 
                static_cast<RangeType>(jumpsize()) - 
                static_cast<RangeType>(1) + 
                static_cast<RangeType>(offset())) /
                static_cast<RangeType>(jumpsize()) ;
    
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
        beginElement(
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
        RangeType nbElementsVar = static_cast<RangeType>(
            nbElements(containerPtr)
        );
        // -1 since we dont like to jump outside
        RangeType nbJumps = ((nbElementsVar > static_cast<RangeType>(offset()))*
                        (nbElementsVar - 
                        static_cast<RangeType>(offset()) - 
                        static_cast<RangeType>(1)) /
                        static_cast<RangeType>(jumpsize()));

        RangeType lastPosition = nbJumps * 
                            static_cast<RangeType>(jumpsize()) +
                             static_cast<RangeType>(offset());
        // -1 since we need the last position
        RangeType neededJumps = nbElementsVar - 
                                static_cast<RangeType>(1) -
                                lastPosition;
        neededJumps *= neededJumps > 0;

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
    -> 
    void
    {
        endElement.set(
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
    ->    
    typename std::enable_if<T==true>::type
    {
        beforeBeginElement.set(
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
        return endElement.test(
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
        return beforeBeginElement.test(
            containerPtr, 
            index, 
            containerSize
        ) || (
            prev(
                containerPtr, 
                indexCopy,
                static_cast<RangeType>(offset()),
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
        auto && nbElem = nbElements(containerPtr);
     //   assert(nbElem >= off);
        const int elems = (
            nbElem -
            static_cast<RangeType>(offset()) + 
            static_cast<RangeType>(jumpsize()) - 
            static_cast<RangeType>(1)
            ) / 
            static_cast<RangeType>(jumpsize());
        return (elems > 0) * elems;
    }
    
//variables
protected:
    OffsetType offset;
    JumpsizeType jumpsize;
    Size containerSize;
    BeginElement beginElement;
    NextElement nextElement;
    EndElement endElement;
    LastElement lastElement;
    PreviousElement previousElement;
    REndElement beforeBeginElement;
} ;


/**
 * @brief This navigator is a concept. It has an offset and a jumpsize.
 */

template<
    typename T_Offset,
    typename T_Jumpsize
>
struct Navigator<
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    T_Offset,
    T_Jumpsize,
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    false>
{
    using ContainerType = deepiterator::details::UndefinedType;
    using ContainerPtr = ContainerType*;
    using ContainerRef = ContainerType&;
    using ComponentType = deepiterator::details::UndefinedType ;
    using ComponentPtr = ComponentType*;
    using JumpsizeType = T_Jumpsize;
    using OffsetType = T_Offset;
    using IndexType = deepiterator::details::UndefinedType ;
    using RangeType = deepiterator::details::UndefinedType ;
    using Size = deepiterator::details::UndefinedType ;
    using BeginElement = deepiterator::details::UndefinedType ;
    using NextElement = deepiterator::details::UndefinedType ;
    using EndElement = deepiterator::details::UndefinedType ;
    using LastElement = deepiterator::details::UndefinedType ;
    using PreviousElement = deepiterator::details::UndefinedType ;
    using REndElement = deepiterator::details::UndefinedType ;
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
        offset(deepiterator::forward<T_Offset_>(offset)),
        jumpsize(deepiterator::forward<T_Jumpsize_>(jumpsize))
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
    deepiterator::Navigator<
        details::UndefinedType,
        details::UndefinedType,
        typename std::decay<T_Offset>::type,
        typename std::decay<T_Jumpsize>::type,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        false>
{
    using OffsetType = typename std::decay<T_Offset>::type;
    using JumpsizeType = typename std::decay<T_Jumpsize>::type;
    typedef deepiterator::Navigator<
        details::UndefinedType,
        details::UndefinedType,
        OffsetType,
        JumpsizeType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        deepiterator::details::UndefinedType,
        false> ResultType;
    return ResultType(
        deepiterator::forward<T_Offset>(offset),
        deepiterator::forward<T_Jumpsize>(jumpsize));
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
        typename TSize= typename _T::Size,
        typename TBeginElement= typename _T::BeginElement,
        typename TNextElement= typename _T::NextElement,
        typename TEndElement= typename _T::EndElement,
        typename TLastElement= typename _T::LastElement,
        typename TPreviousElement= typename _T::PreviousElement,
        typename TREndElement= typename _T::REndElement
    >
    struct NavigatorTemplates
    {
        using ContainerType = TContainerType;
        using OffsetType = TOffsetType;
        using JumpsizeType = TJumpsizeType;
        using IndexType = TIndexType ;
        using RangeType = TRangeType;
        using Size = TSize;
        using BeginElement = TBeginElement;
        using NextElement = TNextElement;
        using EndElement = TEndElement;
        using LastElement = TLastElement;
        using PreviousElement = TPreviousElement;
        using REndElement = TREndElement;
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
    typename T_Component = typename deepiterator::traits::ComponentType<
        T_ContainerNoRef
    >::type,
    typename T_ContainerCategorie = typename deepiterator::traits::ContainerCategory<
        T_ContainerNoRef
    >::type,
    typename T_ContainerSize = typename deepiterator::traits::Size<
        T_ContainerNoRef
    >,
    typename T_Index = typename deepiterator::traits::IndexType<
        T_ContainerNoRef,
        T_ContainerCategorie
    >::type,
    typename T_Range = typename deepiterator::traits::RangeType<
        T_ContainerNoRef,
        T_ContainerCategorie
    >::type,
    typename T_BeginElement = typename deepiterator::traits::navigator::BeginElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_EndElement = typename deepiterator::traits::navigator::EndElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_NextElement = typename deepiterator::traits::navigator::NextElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    typename T_LastElement = typename deepiterator::traits::navigator::LastElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie>,
    typename T_PreviousElement = typename deepiterator::traits::navigator::PreviousElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    typename T_REndElement = typename deepiterator::traits::navigator::REndElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    bool isBidirectional = not std::is_same<
        T_LastElement, 
        deepiterator::details::UndefinedType
    >::value,
     typename = typename std::enable_if< 
        std::is_same<
            deepiterator::Navigator<
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
                T_Offset,
                T_Jumpsize,
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
                deepiterator::details::UndefinedType,
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
deepiterator::Navigator<
    T_ContainerNoRef,
    T_Component,
    T_Offset,
    T_Jumpsize,
    T_Index,
    T_ContainerSize,
    T_Range,
    T_BeginElement,
    T_NextElement,
    T_EndElement,
    T_LastElement,
    T_PreviousElement,
    T_REndElement,
    isBidirectional
>

{ 
    using ResultType = deepiterator::Navigator<
        T_ContainerNoRef,
        T_Component,
        T_Offset,
        T_Jumpsize,
        T_Index,
        T_ContainerSize,
        T_Range,
        T_BeginElement,
        T_NextElement,
        T_EndElement,
        T_LastElement,
        T_PreviousElement,
        T_REndElement,
        isBidirectional
    > ;
        

    auto && result = ResultType(
        deepiterator::forward<T_Offset>(navi.offset),
        deepiterator::forward<T_Jumpsize>(navi.jumpsize)
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
    typename T_Component = typename deepiterator::traits::ComponentType<
        T_ContainerNoRef
    >::type,
    typename T_ContainerCategorie = typename deepiterator::traits::ContainerCategory<
        T_ContainerNoRef
    >::type ,
    
    typename T_ContainerSize = typename deepiterator::traits::Size<
        T_ContainerNoRef
    >::type,
    typename T_Index = typename deepiterator::traits::IndexType<
        T_ContainerNoRef,
        T_ContainerCategorie
    >::type,
    typename T_Range = typename deepiterator::traits::RangeType<
        T_ContainerNoRef,
        T_ContainerCategorie
    >::type,
    typename T_BeginElement = typename deepiterator::traits::navigator::BeginElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >::type,
    typename T_EndElement = typename deepiterator::traits::navigator::EndElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >::type,
    typename T_NextElement = typename deepiterator::traits::navigator::NextElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >::type,
    typename T_LastElement = typename deepiterator::traits::navigator::LastElement<
        T_ContainerNoRef, 
        T_Index, 
        T_ContainerCategorie
    >::type,
    typename T_PreviousElement = typename deepiterator::traits::navigator::PreviousElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >::type,
    typename T_REndElement = typename deepiterator::traits::navigator::REndElement<
        T_ContainerNoRef, 
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >::type,
    bool isBidirectional = not std::is_same<
        T_LastElement, 
        deepiterator::details::UndefinedType
    >::value
    
>
auto 
HDINLINE
makeNavigator(
    T_Offset && offset,
    T_Jumpsize && jumpsize
)
-> 
    deepiterator::Navigator<
        T_ContainerNoRef,
        T_Component,
        T_Offset,
        T_Jumpsize,
        T_Index,
        T_ContainerSize,
        T_Range,
        T_BeginElement,
        T_NextElement,
        T_EndElement,
        T_LastElement,
        T_PreviousElement,
        T_REndElement,
        isBidirectional
    >

{ 
    using ResultType =  deepiterator::Navigator<
        T_ContainerNoRef,
        T_Component,
        T_Offset,
        T_Jumpsize,
        T_Index,
        T_ContainerSize,
        T_Range,
        T_BeginElement,
        T_NextElement,
        T_EndElement,
        T_LastElement,
        T_PreviousElement,
        T_REndElement,
        isBidirectional
    > ;
    auto && result = ResultType(
        deepiterator::forward<T_Offset>(offset),
        deepiterator::forward<T_Jumpsize>(jumpsize)
    );
    
    return result;
    
}

}// namespace deepiterator
