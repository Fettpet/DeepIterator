#pragma once
#include "PIC/Supercell.hpp"
#include "Traits/Accessor/Ahead.hpp"
#include "Traits/Accessor/Behind.hpp"
#include "Traits/Accessor/Equal.hpp"
#include "Traits/Accessor/Get.hpp"
#include "Traits/Navigator/AfterLastElement.hpp"
#include "Traits/Navigator/BeforeFirstElement.hpp"
#include "Traits/Navigator/LastElement.hpp"
#include "Traits/Navigator/NextElement.hpp"
#include "Traits/Navigator/PreviousElement.hpp"
#include "Traits/Navigator/FirstElement.hpp"
namespace hzdr
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
    hzdr::container::categorie::DoublyLinkListLike
    >
{
    HDINLINE
    TComponent&
    operator() (TContainer*, TIndex& idx)
    {
        return *idx;
    }
};    

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
    hzdr::container::categorie::DoublyLinkListLike
    >
{
    HDINLINE
    bool
    operator() (TContainer* con1, TIndex& idx1, TContainer* con2, TIndex& idx2)
    {
        return con1 == con2 && idx1 == idx2;
    }
};

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
    hzdr::container::categorie::DoublyLinkListLike>
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
};



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
    hzdr::container::categorie::DoublyLinkListLike
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
};

} // namespace accessor
    
    
namespace navigator
{

/**
 * @brief Implementation to get the first element. \see FirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange>
struct FirstElement<
    TContainer, 
    TIndex, 
    TRange,
    hzdr::container::categorie::DoublyLinkListLike>
{
    HDINLINE
    void
    operator() (TContainer* container, TIndex& idx, TRange const & range)
    {
        idx = container->first;
        for(auto i=static_cast<TRange>(0); i<range; ++i)
        {
            idx = idx->next;
            if(idx == nullptr) 
                return;
        }
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
    hzdr::container::categorie::DoublyLinkListLike>
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
};
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
    hzdr::container::categorie::DoublyLinkListLike>
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
};

/**
 * @brief Set the iterator to the last element. \see LastElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange>
struct LastElement<
    TContainer,
    TIndex,
    TRange,
    hzdr::container::categorie::DoublyLinkListLike>
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (
        TContainer* containerPtr, 
        TIndex& index, 
        TRange const & offset, 
        TRange const & jumpsize, 
        TSizeFunction& size)
    {
        auto nbElements = size(containerPtr);
        auto jumps = (nbElements % jumpsize) - ((nbElements - offset) % jumpsize); 
        index = containerPtr->last;

        for(TRange i=static_cast<TRange>(0); i<jumps; ++i)
        {
            if(index == nullptr)
                break;
            index = index->previous;
        }
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
struct PreviousElement<
    TContainer,
    TIndex,
    TRange,
    hzdr::container::categorie::DoublyLinkListLike>
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
};

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
    hzdr::container::categorie::DoublyLinkListLike>
{
    template<typename TRangeFunction>
    HDINLINE
    bool
    test (TContainer*, TIndex const & idx, TRange const & offset, TRangeFunction&)
    const
    {
                auto tmp = idx;
        for(TRange i=0; i < offset; ++i)
        {
            if(tmp == nullptr)
                return true;
            tmp = tmp->previous;
        }
        return tmp == nullptr;
    }
    
    template<typename TRangeFunction>
    HDINLINE
    void
    set (TContainer*, TIndex const & idx, TRangeFunction const &)
    const
    {
        idx = nullptr;
    }
};
}
    
} // namespace traits

}// namespace hzdr
