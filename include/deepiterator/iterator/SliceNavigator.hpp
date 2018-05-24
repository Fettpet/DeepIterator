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
#include <cassert>
#include "deepiterator/traits/Traits.hpp"
#include <limits>

namespace deepiterator 
{

 
template<
    int_fast32_t _distance
>
struct Slice
{
public:
    
    constexpr
    auto 
    distance()
    const
    ->
    int_fast32_t
    {
        if(_distance > 0)
        {
            return _distance;
        }
        else if(_distance == 0)
        {
            return std::numeric_limits< int >::max( );
        }
        else
        {
            return -1 * _distance;
        }
    }
    
    constexpr 
    auto 
    from_start()
    const
    ->
    bool
    {
        return _distance >= 0;
    }
    
};

/**
 * \struct SliceNavigator
 @brief This is the default implementation of the SliceNavigator. The task of 
 the navigator is to define the first element, the next element and an after 
 last element. If the navigator is bidirectional it need also a last element, a 
 previous element and a before first element. 
 The navigator has two traits for parallel walking through the container. The 
 first one is T_Offset. This is used to get the distance from the first element 
 of the container to the first element which will be accessed. This trait can be 
 used to map the thread ID (for example offset = threadIdx.x). The second trait 
 is the jumpsize. The jumpsize is the distance between two iterator elements. 
 The number of threads can be mapped on the jumpsize. With this two traits you 
 can go parallel over all elements and touch each element only one times. 
 Additional the SliceNavigator has a slice template parameter. With this 
 template, the container can be restricted. It supports two cases: 1. Counter 
 modus: only the first n-th elements are used to iterate throw; 2. Ignore modus:
 the last n-th elements are ignored. The slice parameter is used after offset 
 and jumpsize. The slice template parameter  has two function:
 bool from_start(): true if Counter Modus, false if Ignore Modus
 int distance(): n for the modies.
 We had three/six traits for the behaviour of the container. The first three 
 traits are 
 1. define the first element of the container,
 2. define a next element of the container,
 3. define a after last element of the container.
 If the navigator is bidirectional three additional traits are needed
 4. define the last element within the container
 5. define a previous element of the container
 6. define a before first element of the container.
 The navigator use this 8 traits to define methodes for parallel iteration 
 though the container.
 
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
 head is operator()(T_Container*, T_Index&, const T_Range). This trait is only needed
 if the navigator is bidirectional. 
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
    typename T_Slice,
    typename T_Index,
    typename T_ContainerSize,
    typename T_Range,
    typename T_BeginElement,
    typename T_NextElement,
    typename T_EndElement,
    typename T_LastElement = deepiterator::details::UndefinedType,
    typename T_PreviousElement = deepiterator::details::UndefinedType,
    typename T_REndElement = deepiterator::details::UndefinedType, 
    bool isBidirectional = false
>
struct SlicedNavigator
{
// define the types 
    using ContainerType = typename std::decay<T_Container>::type;
    using ContainerPtr = ContainerType*;
    using ContainerRef = ContainerType&;
    using ComponentType = T_Component;
    using ComponentPtr = ComponentType*;
    using JumpsizeType = T_Jumpsize;
    using OffsetType = T_Offset;
    using SliceType = T_Slice;
    using IndexType = T_Index;
    using RangeType = T_Range;
    using Size = T_ContainerSize;
    using BeginElement = T_BeginElement;
    using NextElement = T_NextElement;
    using EndElement = T_EndElement;
    using LastElement = T_LastElement;
    using PreviousElement = T_PreviousElement;
    using REndElement = T_REndElement;
    
public:
// the default constructors
    HDINLINE SlicedNavigator() = default;
    HDINLINE SlicedNavigator(SlicedNavigator const &) = default;
    HDINLINE SlicedNavigator(SlicedNavigator &&) = default;
    HDINLINE ~SlicedNavigator() = default;
    HDINLINE SlicedNavigator& operator=(const SlicedNavigator&) = default;
    HDINLINE SlicedNavigator& operator=(SlicedNavigator&&) = default;

    
    /**
     * @brief Set the offset and the jumpsize to the given values
       @param offset the distance from the start to the first element
       @param jumpsize distance between two elements
    */
    HDINLINE
    SlicedNavigator(
            OffsetType && offset, 
            JumpsizeType && jumpsize,
            SliceType && slice
    ):
        cur_pos(static_cast<RangeType>(0)),
        offset(deepiterator::forward<OffsetType>(offset)),
        jumpsize(deepiterator::forward<JumpsizeType>(jumpsize)),
        slice(deepiterator::forward<SliceType>(slice)),
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
        ComponentPtr & componentPtr,
        IndexType & index,
        RangeType distance
    )
    ->
    RangeType    
    {
    
        assert(containerPtr != nullptr); // containerptr should be valid
        // We jump over distance * jumpsize elements
        RangeType const remainingJumpsize = static_cast<RangeType>(nextElement(
            containerPtr,
            componentPtr,
            index,  
            static_cast<RangeType>(jumpsize()) * distance,
            containerSize
        ));
        
        cur_pos += distance;
        RangeType const nbElem = static_cast<RangeType>(size(containerPtr));
        RangeType const distanceSlice = static_cast<RangeType>(slice.distance());
        // we need the distance from the last element to the current index position
        // this is a round up
       
        // 1. We start counting from the begininng and the position is outside
        // the slice.
        if( slice.from_start() && (cur_pos >= distanceSlice))
        {
            // +1 since distanceSlice == 1 looks only at the current position
            return cur_pos - distanceSlice + 1;
        }
        // 2. We ignore the last elements
        else if(not slice.from_start() && (cur_pos > nbElem - distanceSlice))
        {
            return cur_pos + distanceSlice - nbElem;
        }
        // 3. if it is outside the container 
        else 
        {
            return (
                remainingJumpsize + 
                static_cast<RangeType>(jumpsize()) - 
                static_cast<RangeType>(1)
                ) /
                static_cast<RangeType>(jumpsize());
        }
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
        ComponentPtr & componentPtr,
        IndexType & index,
        RangeType distance
    )
    ->
    typename std::enable_if<
        T==true, 
        RangeType
    >::type
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        // We jump over distance * jumpsize elements
        RangeType remainingJumpsize = static_cast<RangeType>(previousElement(
            containerPtr,
            componentPtr,
            index,
            static_cast<RangeType>(jumpsize()) * distance,
            containerSize
        ));
        cur_pos -= distance;
        RangeType const nbElem = static_cast<RangeType>(size(containerPtr));
        RangeType const distanceSlice = static_cast<RangeType>(slice.distance());
        if(remainingJumpsize == 0)
        {
            // we need the distance from the last element to the current index position
            // this is a round up
           
            // 1. We start counting from the begininng and the position is outside
            // the slice.
            if( slice.from_start() && (static_cast<RangeType>(-1) * cur_pos >= distanceSlice))
            {
                return static_cast<RangeType>(-1) * cur_pos - distanceSlice + static_cast<RangeType>(1);
            }
            // 2. We ignore the last elements
            // The cast is nessacary since the container could be empty
            else if(not slice.from_start() && (static_cast<RangeType>(-1) * cur_pos > nbElem - distanceSlice))
            {
                return static_cast<RangeType>(-1) * cur_pos + distanceSlice - nbElem;
            }
            // 3. if it is outside the container 
            else 
            {
                auto indexCopy(index);
                remainingJumpsize = static_cast<RangeType>(previousElement(
                    containerPtr,
                    componentPtr,
                    indexCopy,
                    static_cast<RangeType>(offset()),
                    containerSize
                ));
                return (remainingJumpsize + static_cast<RangeType>(jumpsize()) - static_cast<RangeType>(1)) / static_cast<RangeType>(jumpsize());
            }
        }
        else 
        {
            return (remainingJumpsize + static_cast<RangeType>(jumpsize()) - static_cast<RangeType>(1) + static_cast<RangeType>(offset())) / static_cast<RangeType>(jumpsize());
        }


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
        ComponentPtr & componentPtr,
        IndexType & index
    )
    ->
    void
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        beginElement(
            containerPtr,
            componentPtr,
            index
        );


        nextElement(
            containerPtr,
            componentPtr,
            index,  
            offset(),
            containerSize
        );


        cur_pos = static_cast<RangeType>(0);
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
        ComponentPtr & componentPtr,
        IndexType & index
    )
    ->
    typename std::enable_if<T==true>::type
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        // set to the first element
        if(slice.from_start())
        {     
           begin(
               containerPtr,
               componentPtr,
               index
           );
            // go to the last element
            auto idxCopy = index;
            auto componentPtrCopy = componentPtr;
            auto counter = 0;        
            while(
                next(
                    containerPtr,
                    componentPtrCopy,
                    idxCopy,
                    1u  
                ) == 0 
                && counter <= slice.distance()
            )
            {
                ++counter;
                componentPtr = componentPtrCopy;
                index = idxCopy;
            }
            cur_pos = static_cast<RangeType>(0);
            if(endElement.test(
                containerPtr,
                componentPtr,
                index, 
                containerSize
            ))
            {
                beforeBeginElement.set(
                    containerPtr,
                    componentPtr,
                    index, 
                    containerSize
                );
            }
            
        }
        else 
        {
            lastElement(
                containerPtr,
                componentPtr,
                index,
                containerSize
            );
            previousElement(
                containerPtr,
                componentPtr,
                index,
                slice.distance(),
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
        ComponentPtr & componentPtr,
        IndexType & index
    )
    ->
    void
    {
        endElement.set(
            containerPtr,
            componentPtr,
            index,
            containerSize
        );
    }
    
    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: index of the before first element
     */

    template< bool T=isBidirectional>
    HDINLINE 
    auto
    rend(
        ContainerPtr containerPtr,
        ComponentPtr & componentPtr,
        IndexType & index
    )
    ->
    typename std::enable_if<T==true>::type
    {
        beforeBeginElement.set(
            containerPtr,
            componentPtr,
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
    bool
    isAfterLast(
        ContainerPtr containerPtr,
        ComponentPtr const & componentPtr,
        IndexType const & index)
    const
    {
        
        /*there are three cases: 
         * 1. if the trait say it after the last element
         * 2. if slice.from_start() and cur_pos > slice.distance()
         * 3. not slice.from_start() and cur_pos > nbElements - offset - slice.distance()
         */
        RangeType const distance = static_cast<RangeType>(slice.distance());
        RangeType const nbElem = static_cast<RangeType>(nbElements(containerPtr));
        RangeType const off = static_cast<RangeType>(offset());
        RangeType const jump = static_cast<RangeType>(jumpsize());
        return 
             endElement.test(
                 containerPtr,
                 componentPtr,
                 index, 
                 containerSize
             ) 
            || (slice.from_start() && (cur_pos >= distance))
            || (not slice.from_start() && (cur_pos * jump + off >=
                 nbElem - distance ));
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
        ComponentPtr const & componentPtr,
        IndexType const & index)
    const
    ->
    typename std::enable_if<
        T==true, 
        bool
    >::type
    {
        RangeType const distance = static_cast<RangeType>(slice.distance());
        RangeType const nbElem = static_cast<RangeType>(nbElements(containerPtr)); 
        RangeType const off = static_cast<RangeType>(offset());
        RangeType const jump = static_cast<RangeType>(jumpsize());
        PreviousElement prev(previousElement);
        IndexType indexCopy = index;

        auto componentPtrCopy = componentPtr;
        bool beforeFirst = beforeBeginElement.test(
                    containerPtr,
                    componentPtr,
                    index, 
                    containerSize
                );
        bool prevValue = not beforeFirst && prev(
            containerPtr,
            componentPtrCopy,
            indexCopy,
            offset(),
            containerSize
        ) != 0;
        bool rest = (slice.from_start() && (cur_pos <= static_cast<RangeType>(-1) * distance))
                || (not slice.from_start() && (cur_pos * jump - off) <=
                    static_cast<RangeType>(-1) * (nbElem - distance));
        return beforeFirst || prevValue || rest;
    }
    
    
    /**
     * @brief this function determine the number of elements within the container
     * @param containerPtr pointer to the container, you like to know the number
     * of elements
     * @return number of elements within the container
     */
    HDINLINE
    auto 
    nbElements(ContainerPtr containerPtr)
    const
    ->
    RangeType
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        return containerSize(containerPtr);
    }
    
    
    /**
     * @brief this function determine the number of elements over which the navigator
     * goes. I.e sizeContainer / jumpsize
     * @param containerPtr pointer to the container, you like to know the number
     * of elements
     * @return number of elements the navigator can access
     */
    HDINLINE
    auto
    size(ContainerPtr containerPtr)
    const
    ->
    RangeType 
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        
        RangeType const nbElem = static_cast<RangeType>(nbElements(containerPtr));
        RangeType const off = static_cast<RangeType>(offset());
        RangeType const jump = static_cast<RangeType>(jumpsize());
        RangeType const distance = static_cast<RangeType>(slice.distance());
        if(slice.from_start())
        {
            
            // 1. Case nbElem - off > slice.distance()
            RangeType const sizeFirstCase = (
                    distance
                    + jump
                    - static_cast<RangeType>(1))
                / jump;
            // 2. Case nbElem - off < slice.distance()
            RangeType const sizeSecondCase = (nbElem > off) *(
                    nbElem 
                    - off
                    + jump
                    - static_cast<RangeType>(1)) 
                / jump;

            // check and give it back
            return (nbElem - off >= distance) * sizeFirstCase 
                + (nbElem - off < distance) * sizeSecondCase;
        }
        // it ignores the last slice.distance() elements
        else 
        {
            // 1. Case nbElem - off > slice.distance()
            // I had nbElem - off - slice.distance() elements
                RangeType const sizeFirstCase = (
                    nbElem - off - distance
                    + jump
                    - static_cast<RangeType>(1))
                / jump;
            // 2. Case nbElem - off < slice.distance()
            // I had 0 elements inside
            
                
            return (off < nbElem) * (nbElem - off >= distance) * sizeFirstCase;
                
        }
    }
    
//variables
protected:
    RangeType cur_pos;
    OffsetType offset;
    JumpsizeType jumpsize;
    SliceType slice;
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
    typename T_Jumpsize,
    typename T_Slice>
struct SlicedNavigator<
    deepiterator::details::UndefinedType,
    deepiterator::details::UndefinedType,
    T_Offset,
    T_Jumpsize,
    T_Slice,
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
    using SliceType = T_Slice;
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
    HDINLINE SlicedNavigator() = default;
    HDINLINE SlicedNavigator(SlicedNavigator const &) = default;
    HDINLINE SlicedNavigator(SlicedNavigator &&) = default;
    HDINLINE ~SlicedNavigator() = default;
    
    /**
     * @brief Set the offset and the jumpsize to the given values
       @param offset the distance from the start to the first element
       @param jumpsize distance between two elements
    */
    template<
        typename T_Offset_,
        typename T_Jumpsize_,
        typename T_Slice_>
    HDINLINE
    SlicedNavigator(
            T_Offset_ && offset, 
            T_Jumpsize_ && jumpsize,
            T_Slice_ && slice
             ):
        offset(deepiterator::forward<T_Offset_>(offset)),
        jumpsize(deepiterator::forward<T_Jumpsize_>(jumpsize)),
        slice(deepiterator::forward<T_Slice_>(slice))
    {}
    
    OffsetType offset;
    JumpsizeType jumpsize;
    SliceType slice;
    
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
    typename T_Jumpsize,
    typename T_Slice>
HDINLINE
auto 
makeNavigator(
    T_Offset && offset,
    T_Jumpsize && jumpsize,
    T_Slice && slice
             )
-> 
    deepiterator::SlicedNavigator<
        details::UndefinedType,
        details::UndefinedType,
        typename std::decay<T_Offset>::type,
        typename std::decay<T_Jumpsize>::type,
        typename std::decay<T_Slice>::type,
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
    using OffsetType = typename std::decay<T_Offset>::type ;
    using JumpsizeType = typename std::decay<T_Jumpsize>::type ;
    using SliceType = typename std::decay<T_Slice>::type;
    using ResultType =  deepiterator::SlicedNavigator<
        details::UndefinedType,
        details::UndefinedType,
        OffsetType,
        JumpsizeType,
        SliceType,
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
    >;
    
    auto && result = ResultType(
        deepiterator::forward<T_Offset>(offset),
        deepiterator::forward<T_Jumpsize>(jumpsize),
        deepiterator::forward<T_Slice>(slice)
    );
    return result;
}



namespace details
{

    template<
        typename T,
        typename _T = typename std::decay<T>::type,
        typename T_SliceType = typename _T::SliceType,
        typename T_ContainerType = typename _T::ContainerType,
        typename T_Offset = typename _T::OffsetType,
        typename T_Jumpsize = typename _T::JumpsizeType,
        typename T_Index = typename _T::IndexType,
        typename T_Range = typename _T::RangeType,
        typename TSize = typename _T::Size,
        typename T_BeginElement = typename _T::BeginElement,
        typename T_NextElement = typename _T::NextElement,
        typename T_EndElement = typename _T::EndElement,
        typename TLast = typename _T::LastElement,
        typename TPrevious = typename _T::PreviousElement,
        typename TBeforeFirst = typename _T::REndElement
    >
    struct SlicedNavigatorTemplates
    {
        using ContainerType = T_ContainerType;
        using OffsetType = T_ContainerType;
        using JumpsizeType = T_Jumpsize;
        using SliceType = T_SliceType;
        using IndexType = T_Index;
        using RangeType = T_Range;
        using Size = TSize;
        using BeginElement = T_BeginElement;
        using NextElement = T_NextElement;
        using EndElement = T_EndElement;
        using LastElement = TLast;
        using PreviousElement = TPrevious;
        using REndElement = TBeforeFirst;
    };


template<
    typename T_Container,
    typename T_ContainerNoRef = typename std::decay<T_Container>::type,
    typename OffsetType,
    typename JumpsizeType,
    typename SliceType,
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
        T_Component,
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_EndElement = typename deepiterator::traits::navigator::EndElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_NextElement = typename deepiterator::traits::navigator::NextElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    typename T_LastElement = typename deepiterator::traits::navigator::LastElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_PreviousElement = typename deepiterator::traits::navigator::PreviousElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_Range,
        T_ContainerCategorie
    >,
    typename T_REndElement = typename deepiterator::traits::navigator::REndElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    bool isBidirectional = not std::is_same<
        T_LastElement, 
        deepiterator::details::UndefinedType
    >::value
>
auto
HDINLINE
makeNavigator(
    deepiterator::SlicedNavigator<
        details::UndefinedType,
        details::UndefinedType,
        OffsetType,
        JumpsizeType,
        SliceType,
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
    > & navi
)
->
deepiterator::SlicedNavigator<
    T_ContainerNoRef,
    T_Component,
    OffsetType,
    JumpsizeType,
    SliceType,
    T_Index,
    T_ContainerSize,
    T_Range,
    T_BeginElement,
    T_NextElement,
    T_EndElement,
    T_LastElement,
    T_PreviousElement,
    T_REndElement,
    isBidirectional>
{
    using ResultType = deepiterator::SlicedNavigator<
        T_ContainerNoRef,
        T_Component,
        OffsetType,
        JumpsizeType,
        SliceType,
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
    >;    

    auto && result = ResultType(
        deepiterator::forward<OffsetType>(navi.offset), 
        deepiterator::forward<JumpsizeType>(navi.jumpsize),
        deepiterator::forward<SliceType>(navi.slice)
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
    typename T_Slice,
    typename T_Component = typename deepiterator::traits::ComponentType<
        T_ContainerNoRef
    >::type,
    typename T_ContainerCategorie = typename deepiterator::traits::ContainerCategory<
        T_ContainerNoRef
    >::type,
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
        T_Component,
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_EndElement = typename deepiterator::traits::navigator::EndElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_NextElement = typename deepiterator::traits::navigator::NextElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    typename T_LastElement = typename deepiterator::traits::navigator::LastElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_ContainerCategorie
    >,
    typename T_PreviousElement = typename deepiterator::traits::navigator::PreviousElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_Range,
        T_ContainerCategorie
    >,
    typename T_REndElement = typename deepiterator::traits::navigator::REndElement<
        T_ContainerNoRef,
        T_Component,
        T_Index, 
        T_Range, 
        T_ContainerCategorie
    >,
    bool isBidirectional = not std::is_same<
        T_LastElement, 
        deepiterator::details::UndefinedType
    >::value
>
auto 
HDINLINE
makeNavigator(
    T_Offset && offset,
    T_Jumpsize && jumpsize,
    T_Slice && slice
             )
-> 
    deepiterator::SlicedNavigator<
        T_ContainerNoRef,
        T_Component,
        typename std::decay<T_Offset>::type,
        typename std::decay<T_Jumpsize>::type,
        typename std::decay<T_Slice>::type,
        T_Index,
        T_ContainerSize,
        T_Range,
        T_BeginElement,
        T_NextElement,
        T_EndElement,
        T_LastElement,
        T_PreviousElement,
        T_REndElement,
        isBidirectional>
{

    using ResultType = deepiterator::SlicedNavigator<
        T_ContainerNoRef,
        T_Component,
        typename std::decay<T_Offset>::type,
        typename std::decay<T_Jumpsize>::type,
        typename std::decay<T_Slice>::type,
        T_Index,
        T_ContainerSize,
        T_Range,
        T_BeginElement,
        T_NextElement,
        T_EndElement,
        T_LastElement,
        T_PreviousElement,
        T_REndElement,
        isBidirectional>;
        
    auto && result = ResultType(
        deepiterator::forward<T_Offset>(offset),
        deepiterator::forward<T_Jumpsize>(jumpsize),
        deepiterator::forward<T_Slice>(slice)
    );
    
    return result;
}

}// namespace deepiterator

