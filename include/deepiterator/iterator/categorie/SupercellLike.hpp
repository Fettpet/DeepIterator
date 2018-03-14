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


/**
* @brief This file contains traits for a supercell like datatype. A supercell is 
a double linked list. It has two public variables firstFrame and lastFrame. Both
are pointer. Both variables are class like data structures and have two public 
variables: nextFrame and previousFrame. Both are pointer. These are the nodes of
the linked list. The following conditions must hold
1. container.firstFrame->previousFrame == nullptr
2. container.lastFrame->nextFrame == nullptr
3. frame->nextFrame->previousFrame == frame, if frame != nullptr and frame->nextFrame
 != nullptr
4. frame->previousFrame->nextFrame == frame, if frame != nullptr and frame->previousFrame
 != nullptr
*/
#pragma once

#include "deepiterator/traits/Traits.hpp"
#include "deepiterator/definitions/hdinline.hpp"
namespace hzdr
{

namespace container 
{
namespace categorie
{

struct SupercellLike;


} // namespace categorie

} // namespace container

namespace traits 
{
template<typename TContainer>
struct IsBidirectional<
    TContainer, 
    hzdr::container::categorie::SupercellLike
>
{
    static const bool value = true;
} ;    

template<typename TContainer>
struct IsRandomAccessable<
    TContainer, 
    hzdr::container::categorie::SupercellLike
>
{
    static const bool value = true;
} ;
namespace accessor
{

/**
 * @brief get the value of the element, at the iterator positions. \see Get.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex
>
struct Get<
    TContainer,
    TComponent, 
    TIndex, 
    hzdr::container::categorie::SupercellLike
>
{
    HDINLINE
    TComponent&
    operator() (TContainer*, TIndex& idx)
    {
        return *idx;
    }
} ;    

/**
 * @brief check if both iterators are at the same element. \see Equal.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex
>
struct Equal<
    TContainer,
    TComponent, 
    TIndex, 
    hzdr::container::categorie::SupercellLike
>
{
    HDINLINE
    bool
    operator() (
        TContainer* con1, 
        TIndex const & idx1, 
        TContainer* con2, 
        TIndex const & idx2
    )
    {
        return con1 == con2 && idx1 == idx2;
    }
} ;

 /**
 * @brief Check if the iterator one is ahead the second one. \see Ahead.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex
>
struct Ahead<
    TContainer,
    TComponent, 
    TIndex, 
    hzdr::container::categorie::SupercellLike
>
{
    HDINLINE
    bool
    operator() (
        TContainer* con1, 
        TIndex const & idx1, 
        TContainer* con2, 
        TIndex const & idx2
    )
    {
        if(con1 != con2)
            return false;
        
        TIndex tmp = idx1;
        while(tmp != nullptr)
        {
            tmp = tmp->previousFrame.ptr;
            if(tmp == idx2) 
                return true;
           
        }
        return false;
    }
} ;



/**
 * @brief check wheter the iterator 1 is behind the second one. \see Behind.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex
>
struct Behind<
    TContainer,
    TComponent, 
    TIndex, 
    hzdr::container::categorie::SupercellLike
>
{
    HDINLINE
    bool
    operator() (
        TContainer*, 
        TIndex const & idx1, 
        TContainer*, 
        TIndex const & idx2
    )
    {
        TIndex tmp = idx1;
        while(tmp != nullptr)
        {
            tmp = tmp->nextFrame.ptr;
            if(tmp == idx2) 
                return true;
            
        }
        return false;
    }
} ;

} // namespace accessor
    
    
namespace navigator
{

/**
 * @brief Implementation to get the first element. \see FirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex
>
struct FirstElement<
    TContainer,
    TIndex, 
    hzdr::container::categorie::SupercellLike
>
{
    HDINLINE
    void
    operator() (
        TContainer* container, 
        TIndex & idx
    )
    {
        idx = container->firstFramePtr;
    }
} ;
/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange
>
struct NextElement<
    TContainer,
    TIndex,
    TRange,
    hzdr::container::categorie::SupercellLike
>
{

    template<
        typename TContainerSize
    >
    HDINLINE
    TRange
    operator() (
        TContainer*, 
        TIndex& idx, 
        TRange const & range,
        TContainerSize&)
    {
        TRange i = 0;
        for(i = 0; i<range; ++i)
        {
            idx = idx->nextFrame.ptr;
            if(idx == nullptr)
                break;
        }
        return range - i;
    }
} ;
/**
 * @brief Implementation to check whether the iterator is after the last element.
 * \see AfterLastElement.hpp
 */
template<
    typename TContainer,
    typename TIndex
>
struct AfterLastElement<
    TContainer,
    TIndex, 
    hzdr::container::categorie::SupercellLike
>
{
    template<typename TRangeFunction>
    HDINLINE
    bool
    test(
        TContainer*, 
        TIndex const & idx, 
        TRangeFunction const &
    )
    const
    {
        return idx == nullptr || idx.ptr == nullptr;
    }
    
    template<typename TRangeFunction>
    HDINLINE
    void
    set(
        TContainer*, 
        TIndex & idx,
        TRangeFunction const &
    )
    const
    {
        idx = nullptr;
    }
} ;

/**
 * @brief Set the iterator to the last element. \see LastElement.hpp
 */
template<
    typename TContainer,
    typename TIndex
>
struct LastElement<
    TContainer,
    TIndex,
    hzdr::container::categorie::SupercellLike
>
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (
        TContainer* containerPtr, 
        TIndex& index, 
        TSizeFunction &&
    )
    {
        index = containerPtr->lastFramePtr;
    }
} ;

/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TIndex,
    typename TContainer,
    typename TRange
>
struct PreviousElement<
    TContainer,
    TIndex,
    TRange,
    hzdr::container::categorie::SupercellLike
>
{
    
    template<
        typename TContainerSize>
    HDINLINE
    TRange
    operator() (
        TContainer*, 
        TIndex& idx, 
        TRange const & jumpsize,
        TContainerSize&)
    {
        TRange i = 0;
        for(i = 0; i<jumpsize; ++i)
        {
            idx = idx->previousFrame.ptr;
            if(idx == nullptr)
                return jumpsize - i;
        }

        return jumpsize - i;
    }
} ;

/**
 * @brief Implementation to check whether the iterator is before the fist 
 * element. \see BeforeFirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange
>
struct BeforeFirstElement<
    TContainer,
    TIndex, 
    TRange,
    hzdr::container::categorie::SupercellLike
>
{
    
    template<typename TRangeFunction>
    HDINLINE
    bool
    test(
        TContainer*, 
        TIndex const & idx,
        TRangeFunction&
    )
    const
    {
        return idx == nullptr || idx.ptr == nullptr;
    }
    

    template<typename TRangeFunction>
    HDINLINE
    void
    set(
        TContainer*, 
        TIndex & idx,
        TRangeFunction&
    )
    const
    {
        idx = nullptr;
    }
} ;
}
    
} // namespace traits

}// namespace hzdr

