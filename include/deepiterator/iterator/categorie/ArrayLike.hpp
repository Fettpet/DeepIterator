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
struct ArrayLike;



} //namespace container

} // namespace Categorie

// implementation of the traits
namespace traits
{
template<typename TContainer>
struct IsBidirectional<
    TContainer, 
    deepiterator::container::categorie::ArrayLike>
{
    static const bool value = true;
} ;    

template<typename TContainer>
struct IsRandomAccessable<
    TContainer, 
    deepiterator::container::categorie::ArrayLike>
{
    static const bool value = true;
} ;

template<
    typename TContainer
>
struct RangeType<TContainer, deepiterator::container::categorie::ArrayLike>
{
    typedef int_fast32_t type;
};

template<
    typename TContainer
>
struct IndexType<TContainer, deepiterator::container::categorie::ArrayLike>
{
    typedef int_fast32_t type;
};

namespace accessor 
{
/**
 * @brief Get the value out of the container at the current iterator position.
 * \see Get.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex>
struct At<
    TContainer, 
    TComponent, 
    TIndex, 
    deepiterator::container::categorie::ArrayLike
    >
{
    HDINLINE
    TComponent&
    operator() (
        TContainer* con,
        TComponent* &,
        TIndex const & idx
    )
    {
        // is not implemented. Specify the trait
        return (*con)[idx];
    }
    
} ;

/**
 * @brief check whether to iterators are at the same position. \see Equal.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex>
struct Equal<
    TContainer, 
    TComponent, 
    TIndex, 
    deepiterator::container::categorie::ArrayLike
    >
{
    HDINLINE
    bool
    operator() (
        TContainer * const con1,
        TComponent * const,
        TIndex const & idx1, 
        TContainer * const con2,
        TComponent * const,
        TIndex const & idx2)
    {
        // is not implemented. Specify the trait
        return con1 == con2 && idx1 == idx2;
    }
    
} ;

/**
 * @brief check whether the first iterator is ahead the second one. 
 * \see Ahead.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex>
struct Ahead<
    TContainer, 
    TComponent, 
    TIndex, 
    deepiterator::container::categorie::ArrayLike>
{
    HDINLINE
    bool
    operator() (
        TContainer * const con1,
        TComponent * const,
        TIndex const & idx1, 
        TContainer * const con2,
        TComponent * const,
        TIndex const & idx2)
    {
        // is not implemented. Specify the trait
        return idx1 > idx2 && con1 == con2;
    }
    
} ;

/**
 * @brief check whether the first iterator is behind the first one. 
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex>
struct Behind<
    TContainer, 
    TComponent, 
    TIndex, 
    deepiterator::container::categorie::ArrayLike
    >
{
    HDINLINE
    bool
    operator() (
        TContainer * const,
        TComponent * const,
        TIndex const & idx1, 
        TContainer *,
        TComponent * const,
        TIndex const & idx2)
    {
        // is not implemented. Specify the trait
        return idx1 < idx2;
    }
    
} ;
    
} // namespace accessor
    
    
    
    
namespace navigator
{
/**
 * @brief implementation to get the first element within a container. For further
 * details \see BeginElement.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex>
struct BeginElement<
    TContainer,
    TComponent,
    TIndex, 
    deepiterator::container::categorie::ArrayLike>
{
    HDINLINE
    void
    operator() (
        TContainer * ,
        TComponent * & componentPtr,
        TIndex& idx)
    {
        componentPtr = nullptr;
        idx = static_cast<TIndex>(0);
    }
    
} ;

/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TRange>
struct NextElement<
    TContainer,
    TComponent,
    TIndex,
    TRange,
    deepiterator::container::categorie::ArrayLike>
{
    template<
        typename TContainerSize>
    HDINLINE
    TRange
    operator() (
        TContainer * container,
        TComponent * &,
        TIndex & idx,
        TRange const & range,
        TContainerSize & size)
    {
        idx += range;
        return (idx >= static_cast<TRange>(size(container))) * (idx - (static_cast<TRange>(size(container))-1) );
    }
    
} ;

/**
 * @brief Implementation to check whether the end is reached. For further 
 * informations \see EndElement.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex
>
struct EndElement<
    TContainer,
    TComponent,
    TIndex, 
    deepiterator::container::categorie::ArrayLike
>
{
    template<typename TSizeFunction>
    HDINLINE
    bool
    test (
        TContainer * conPtr,
        TComponent * const &,
        TIndex const & idx, 
        TSizeFunction const & size)
    const
    {
        return idx >= size(conPtr);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set(
        TContainer* conPtr,
        TComponent * &,
        TIndex & idx,
        TSizeFunction const & size)
    const
    {
        idx = size(conPtr);
    }
    
} ;

/**
 * @brief Implementation of the array like last element trait. For further details
 * \see LastElement.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex
>
struct LastElement<
    TContainer,
    TComponent,
    TIndex,
    deepiterator::container::categorie::ArrayLike>
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (
        TContainer * conPtr,
        TComponent * &,
        TIndex& index,
        TSizeFunction& size
    )
    {
        index = size(conPtr) - 1;
    }
    
} ;

/**
 * @brief The implementation to get the last element in a array like data
 * structure. For futher details \see PreviousElement.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TRange
>
struct PreviousElement<
    TContainer,
    TComponent,
    TIndex,
    TRange,
    deepiterator::container::categorie::ArrayLike
>
{
    template<typename T>
    HDINLINE
    int
    operator() (
        TContainer *,
        TComponent * &,
        TIndex& idx, 
        TRange const & jumpsize,
        T const &
    )
    {
        idx -= jumpsize;

        return (static_cast<int>(idx) < 0) * (-1 * static_cast<int>(idx));
    }
    
} ;

/**
 * @brief Implmentation to get check whether the iterator is on the element 
 * before the first one. \see REndElement.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TOffset
>
struct REndElement<
    TContainer,
    TComponent,
    TIndex,
    TOffset,
    deepiterator::container::categorie::ArrayLike
>
{
    template<typename TSizeFunction>
    HDINLINE
    bool
    test(
        TContainer*,
        TComponent* const &,
        TIndex const & idx,
        TSizeFunction&
    )
    const
    {
        return static_cast<int>(idx) < static_cast<int>(0);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set(
        TContainer*,
        TComponent* & component,
        TIndex & idx,
        TSizeFunction&
    )
    const
    {
        component = nullptr;
        idx = static_cast<TIndex>(-1);
    }
    
} ;

    
}// namespace navigator
} // namespace traits
    
}// namespace deepiterator
