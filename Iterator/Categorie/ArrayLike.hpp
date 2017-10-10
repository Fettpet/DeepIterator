#pragma once
#include "Traits/Accessor/Ahead.hpp"
#include "Traits/Accessor/Behind.hpp"
#include "Traits/Accessor/Equal.hpp"
#include "Traits/Accessor/Get.hpp"
#include "Traits/Navigator/AfterLastElement.hpp"
#include "Traits/Navigator/BeforeFirstElement.hpp"
#include "Traits/Navigator/LastElement.hpp"
#include "Traits/Navigator/NextElement.hpp"
#include "Traits/Navigator/PreviousElement.hpp"

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
    operator() (TContainer* con, TIndex& idx)
    {
        // is not implemented. Specify the trait
        return (*con)[idx];
    }
};

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
    operator() (TContainer* con1, TIndex& idx1, TContainer* con2, TIndex& idx2)
    {
        // is not implemented. Specify the trait
        return con1 == con2 && idx1 == idx2;
    }
};

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
    operator() (TContainer* con1, TIndex& idx1, TContainer* con2, TIndex& idx2)
    {
        // is not implemented. Specify the trait
        return idx1 > idx2 && con1 == con2;
    }
};

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
    operator() (TContainer*, TIndex& idx1, TContainer*, TIndex& idx2)
    {
        // is not implemented. Specify the trait
        return idx1 < idx2;
    }
};
    
} // namespace accessor
    
    
    
    
namespace navigator
{
/**
 * @brief implementation to get the first element within a container. For further
 * details \see FirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange>
struct FirstElement<
    TContainer, 
    TIndex, 
    TRange,
    hzdr::container::categorie::ArrayLike>
{
    HDINLINE
    void
    operator() (TContainer*, TIndex& idx, TRange const & range)
    {
        idx = static_cast<TIndex>(range);
    }
};

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
        return (idx >= size(container)) * (idx - (size(container)-1) );
    }
};

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
    test (TContainer* conPtr, TIndex const & idx, TSizeFunction const & size)
    const
    {
        return idx >= size(conPtr);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set(TContainer* conPtr, TIndex const & idx, TSizeFunction const & size)
    const
    {
        idx = size(conPtr);
    }
    
};

/**
 * @brief Implementation of the array like last element trait. For further details
 * \see LastElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange>
struct LastElement<
    TContainer,
    TIndex,
    TRange,
    hzdr::container::categorie::ArrayLike>
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (TContainer* conPtr, 
                TIndex& index, 
                TRange const & offset, 
                TRange const & jumpsize, 
                TSizeFunction& size)
    {
        auto nbOfJumps = ((size(conPtr) - offset - 1) / jumpsize );
        
        index = (nbOfJumps) * jumpsize + offset;
    }
};

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
    HDINLINE
    void
    operator() (TContainer*, TIndex& idx, TRange& jumpsize)
    {
        idx -= jumpsize;
    }
};

/**
 * @brief Implmentation to get check whether the iterator is on the element 
 * before the first one. \see BeforeFirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange>
struct BeforeFirstElement<
    TContainer, 
    TIndex, 
    TRange,
    hzdr::container::categorie::ArrayLike>
{
    template<typename TRangeFunction>
    HDINLINE
    bool
    test (TContainer*, TIndex const & idx, TRangeFunction&)
    const
    {
        return idx < static_cast<TIndex>(0);
    }
    
    template<typename TRangeFunction>
    HDINLINE
    void
    set (TContainer*, TIndex const & idx, TRangeFunction&)
    const
    {
        idx = static_cast<TIndex>(-1);
    }
};

    
}// namespace navigator
} // namespace traits
    
}// namespace hzdr
