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

#include "deepiterator/traits/Traits.hpp"
#include "deepiterator/definitions/hdinline.hpp"
namespace deepiterator
{
    
namespace container 
{
namespace categorie
{

struct DoublyLinkListLike;


} // namespace categorie

} // namespace contaienr

namespace traits 
{
    
template<typename TContainer>
struct IsBidirectional<
    TContainer, 
    deepiterator::container::categorie::DoublyLinkListLike>
{
    static const bool value = true;
} ;       

template<typename TContainer>
struct IsRandomAccessable<
    TContainer, 
    deepiterator::container::categorie::DoublyLinkListLike>
{
    static const bool value = true;
} ;   

template<typename TContainer>
struct RangeType<
    TContainer, 
    deepiterator::container::categorie::DoublyLinkListLike
>
{
    typedef int_fast32_t type;
};


namespace accessor
{

/**
 * @brief get the value of the element, at the iterator positions. \see Get.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex>
struct Get<
    TContainer, 
    TComponent, 
    TIndex, 
    deepiterator::container::categorie::DoublyLinkListLike
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
    typename TIndex>
struct Equal<
    TContainer, 
    TComponent, 
    TIndex, 
    deepiterator::container::categorie::DoublyLinkListLike
    >
{
    HDINLINE
    bool
    operator() (TContainer* con1, TIndex& idx1, TContainer* con2, TIndex& idx2)
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
    typename TIndex>
struct Ahead<
    TContainer, 
    TComponent, 
    TIndex, 
    deepiterator::container::categorie::DoublyLinkListLike>
{
    HDINLINE
    bool
    operator() (TContainer* con1, TIndex& idx1, TContainer* con2, TIndex& idx2)
    {
        if(con1 != con2)
            return false;
        
        TIndex tmp = idx1;
        while(tmp != nullptr)
        {
            if(tmp == idx2) 
                return true;
            tmp = tmp->previous;
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
    typename TIndex>
struct Behind<
    TContainer, 
    TComponent, 
    TIndex, 
    deepiterator::container::categorie::DoublyLinkListLike
    >
{
    HDINLINE
    bool
    operator() (TContainer*, TIndex& idx1, TContainer*, TIndex& idx2)
    {
        TIndex tmp = idx1;
        while(tmp != nullptr)
        {
            if(tmp == idx2) 
                return true;
            tmp = tmp->next;
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
    typename TIndex>
struct FirstElement<
    TContainer, 
    TIndex, 
    deepiterator::container::categorie::DoublyLinkListLike>
{
    HDINLINE
    void
    operator() (TContainer* container, TIndex& idx)
    {
        idx = container->first;

    }
} ;   
/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange>
struct NextElement<
    TContainer,
    TIndex,
    TRange,
    deepiterator::container::categorie::DoublyLinkListLike>
{
    template<
        typename TContainerSize>
    HDINLINE
    TRange
    operator() (
        TContainer* container, 
        TIndex& idx, 
        TRange const & range,
        TContainerSize& size)
    {
        TRange i = 0;
        for(i = 0; i<range; ++i)
        {
            idx = idx->next;
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
    typename TIndex>
struct AfterLastElement<
    TContainer, 
    TIndex, 
    deepiterator::container::categorie::DoublyLinkListLike>
{
    template<typename TRangeFunction>
    HDINLINE
    bool
    test (TContainer*, TIndex const & idx, TRangeFunction const &)
    const
    {
        return idx == nullptr;
    }
    
    template<typename TRangeFunction>
    HDINLINE
    void
    set (TContainer*, TIndex const & idx, TRangeFunction const &)
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
    typename TIndex>
struct LastElement<
    TContainer,
    TIndex,
    deepiterator::container::categorie::DoublyLinkListLike>
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (
        TContainer* containerPtr, 
        TIndex& index, 
        TSizeFunction& size)
    {
        index = containerPtr->last;

    }
} ;   

/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange>
struct PreviousElement<
    TContainer,
    TIndex,
    TRange,
    deepiterator::container::categorie::DoublyLinkListLike>
{
    template<
        typename TContainerSize>
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
            idx = idx->PreviousElement;
            if(idx == nullptr)
                break;
        }
        return range - i;
    }
} ;   

/**
 * @brief Implementation to check whether the iterator is before the fist 
 * element. \see BeforeFirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange>
struct BeforeFirstElement<
    TContainer, 
    TIndex, 
    TRange,
    deepiterator::container::categorie::DoublyLinkListLike>
{
    template<typename TRangeFunction>
    HDINLINE
    auto
    test (
        TContainer*, 
        TIndex const & idx,
        TRangeFunction&
    )
    const
    ->
    bool
    {

        return idx == nullptr;
    }
    
    template<typename TRangeFunction>
    HDINLINE
    auto
    set (TContainer*, TIndex const & idx, TRangeFunction const &)
    const
    ->
    void
    {
        idx = nullptr;
    }
} ;   
}
    
} // namespace traits

}// namespace deepiterator
