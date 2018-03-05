#pragma once

#include "deepiterator/traits/Traits.hpp"
#include "deepiterator/definitions/hdinline.hpp"
namespace hzdr
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
    hzdr::container::categorie::ArrayLike>
{
    static const bool value = true;
} ;    

template<typename TContainer>
struct IsRandomAccessable<
    TContainer, 
    hzdr::container::categorie::ArrayLike>
{
    static const bool value = true;
} ;

template<
    typename TContainer
>
struct RangeType<TContainer, hzdr::container::categorie::ArrayLike>
{
    typedef int_fast32_t type;
};

template<
    typename TContainer
>
struct IndexType<TContainer, hzdr::container::categorie::ArrayLike>
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
struct Get<
    TContainer, 
    TComponent, 
    TIndex, 
    hzdr::container::categorie::ArrayLike
    >
{
    HDINLINE
    TComponent&
    operator() (TContainer* con, TIndex const & idx)
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
    hzdr::container::categorie::ArrayLike
    >
{
    HDINLINE
    bool
    operator() (
        TContainer * const con1, 
        TIndex const & idx1, 
        TContainer * const con2, 
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
    hzdr::container::categorie::ArrayLike>
{
    HDINLINE
    bool
    operator() (
        TContainer * const con1, 
        TIndex const & idx1, 
        TContainer * const con2, 
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
    hzdr::container::categorie::ArrayLike
    >
{
    HDINLINE
    bool
    operator() (
        TContainer * const, 
        TIndex const & idx1, 
        TContainer *, 
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
 * details \see FirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex>
struct FirstElement<
    TContainer, 
    TIndex, 
    hzdr::container::categorie::ArrayLike>
{
    HDINLINE
    void
    operator() (TContainer*, TIndex& idx)
    {
        idx = static_cast<TIndex>(0);
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
    hzdr::container::categorie::ArrayLike>
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
        idx += range;
        return (idx >= static_cast<TRange>(size(container))) * (idx - (static_cast<TRange>(size(container))-1) );
    }
    
} ;

/**
 * @brief Implementation to check whether the end is reached. For further 
 * informations \see AfterLastElement.hpp
 */
template<
    typename TContainer,
    typename TIndex>
struct AfterLastElement<
    TContainer, 
    TIndex, 
    hzdr::container::categorie::ArrayLike>
{
    template<typename TSizeFunction>
    HDINLINE
    bool
    test (
        TContainer* conPtr, 
        TIndex const & idx, 
        TSizeFunction const & size)
    const
    {
        return idx >= size(conPtr);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set(TContainer* conPtr, TIndex & idx, TSizeFunction const & size)
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
    typename TIndex>
struct LastElement<
    TContainer,
    TIndex,
    hzdr::container::categorie::ArrayLike>
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (TContainer* conPtr, 
                TIndex& index, 
                TSizeFunction& size)
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
    typename TIndex,
    typename TRange>
struct PreviousElement<
    TContainer,
    TIndex,
    TRange,
    hzdr::container::categorie::ArrayLike>
{
    template<typename T>
    HDINLINE
    int
    operator() (
        TContainer*, 
        TIndex& idx, 
        TRange const & offset,
        TRange const & jumpsize,
        T const &
               )
    {
        idx -= jumpsize;
        // only return if idx < offset
        return (static_cast<int>(idx) < static_cast<int>(offset)) * (static_cast<int>(offset) - static_cast<int>(idx));
    }
    
} ;

/**
 * @brief Implmentation to get check whether the iterator is on the element 
 * before the first one. \see BeforeFirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TOffset>
struct BeforeFirstElement<
    TContainer, 
    TIndex,
    TOffset,
    hzdr::container::categorie::ArrayLike>
{
    template<typename TSizeFunction>
    HDINLINE
    bool
    test (TContainer*, TIndex const & idx, TOffset const & offset, TSizeFunction&)
    const
    {
        return static_cast<int>(idx) < static_cast<int>(offset);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set (TContainer*, TIndex & idx, TOffset const &, TSizeFunction&)
    const
    {
        idx = static_cast<TIndex>(-1);
    }
    
} ;

    
}// namespace navigator
} // namespace traits
    
}// namespace hzdr
