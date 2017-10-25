#pragma once
#include "PIC/Supercell.hpp"
#include "Traits/IsBidirectional.hpp"
#include "Traits/IsRandomAccessable.hpp"
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
namespace traits 
{
template<typename SFIANE, typename TFrame>
struct IsBidirectional<
    hzdr::Supercell<TFrame>, 
    SFIANE>
{
    static const bool value = true;
};    

template<typename SFIANE, typename TFrame>
struct IsRandomAccessable<
    hzdr::Supercell<TFrame>, 
    SFIANE>
{
    static const bool value = true;
};
namespace accessor
{

/**
 * @brief get the value of the element, at the iterator positions. \see Get.hpp
 */
template<
    typename TContainerCategorie,
    typename TFrame,
    typename TComponent,
    typename TIndex>
struct Get<
    hzdr::Supercell<TFrame>,
    TComponent, 
    TIndex, 
    TContainerCategorie
    >
{
    typedef hzdr::Supercell<TFrame> TContainer;
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
    typename TFrame,
    typename TComponent,
    typename TContainerCategorie,
    typename TIndex>
struct Equal<
    hzdr::Supercell<TFrame>,
    TComponent, 
    TIndex, 
    TContainerCategorie
    >
{
    typedef hzdr::Supercell<TFrame> TContainer;
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
    typename TFrame,
    typename TComponent,
    typename TContainerCategorie,
    typename TIndex>
struct Ahead<
    hzdr::Supercell<TFrame>,
    TComponent, 
    TIndex, 
    TContainerCategorie>
{
    typedef hzdr::Supercell<TFrame> TContainer;
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
            tmp = tmp->previousFrame;
        }
        return false;
    }
};



/**
 * @brief check wheter the iterator 1 is behind the second one. \see Behind.hpp
 */
template<
    typename TFrame,
    typename TComponent,
    typename TContainerCategorie,
    typename TIndex>
struct Behind<
    hzdr::Supercell<TFrame>,
    TComponent, 
    TIndex, 
    TContainerCategorie
    >
{
    typedef hzdr::Supercell<TFrame> TContainer;
    HDINLINE
    bool
    operator() (TContainer*, TIndex& idx1, TContainer*, TIndex& idx2)
    {
        TIndex tmp = idx1;
        while(tmp != nullptr)
        {
            if(tmp == idx2) 
                return true;
            tmp = tmp->nextFrame;
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
    typename TFrame,
    typename TIndex,
    typename TContainerCategorie>
struct FirstElement<
    hzdr::Supercell<TFrame>,
    TIndex, 
    TContainerCategorie>
{
    typedef hzdr::Supercell<TFrame> TContainer;
    HDINLINE
    void
    operator() (TContainer* container, TIndex& idx)
    {
        idx = container->firstFrame;
    }
};
/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TFrame,
    typename TIndex,
    typename TContainerCategorie,
    typename TRange>
struct NextElement<
    hzdr::Supercell<TFrame>,
    TIndex,
    TRange,
    TContainerCategorie>
{
    typedef hzdr::Supercell<TFrame> TContainer;
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
            idx = idx->nextFrame;
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
    typename TFrame,
    typename TContainerCategorie,
    typename TIndex>
struct AfterLastElement<
    hzdr::Supercell<TFrame>,
    TIndex, 
    TContainerCategorie>
{
    typedef hzdr::Supercell<TFrame> TContainer;
    template<typename TRangeFunction>
    HDINLINE
    bool
    test(TContainer*, TIndex const & idx, TRangeFunction const &)
    const
    {
        return idx == nullptr;
    }
    
    template<typename TRangeFunction>
    HDINLINE
    void
    set(TContainer*, TIndex & idx, TRangeFunction const &)
    const
    {
        idx = nullptr;
    }
};

/**
 * @brief Set the iterator to the last element. \see LastElement.hpp
 */
template<
    typename TFrame,
    typename TIndex,
    typename TContainerCategorie>
struct LastElement<
    hzdr::Supercell<TFrame>,
    TIndex,
    TContainerCategorie>
{
    typedef hzdr::Supercell<TFrame> TContainer;
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (
        TContainer* containerPtr, 
        TIndex& index, 
        TSizeFunction &&)
    {

        index = containerPtr->lastFrame;


    }
};

/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TFrame,
    typename TIndex,
    typename TContainerCategorie,
    typename TRange>
struct PreviousElement<
    hzdr::Supercell<TFrame>,
    TIndex,
    TRange,
    TContainerCategorie>
{
    
    typedef hzdr::Supercell<TFrame> TContainer;
    template<
        typename TContainerSize>
    HDINLINE
    TRange
    operator() (
        TContainer*, 
        TIndex& idx, 
        TRange const &,
        TRange const & jumpsize,
        TContainerSize&)
    {
        TRange i = 0;
        for(i = 0; i<jumpsize; ++i)
        {
            idx = idx->previousFrame;
            if(idx == nullptr)
                return jumpsize - i;
        }

        return jumpsize - i;
    }
};

/**
 * @brief Implementation to check whether the iterator is before the fist 
 * element. \see BeforeFirstElement.hpp
 */
template<
    typename TFrame,
    typename TIndex,
    typename TContainerCategorie,
    typename TRange>
struct BeforeFirstElement<
    hzdr::Supercell<TFrame>,
    TIndex, 
    TRange,
    TContainerCategorie>
{
    typedef hzdr::Supercell<TFrame> TContainer;
    
    template<typename TRangeFunction>
    HDINLINE
    bool
    test(TContainer*, TIndex const & idx, TRange const & offset, TRangeFunction&)
    const
    {
        TIndex tmp = idx;
        for(TRange i = static_cast<TRange>(0); i < offset; ++i)
        { 
            if(tmp == nullptr)
                return true;
            tmp = tmp->previousFrame;
        }
        return tmp == nullptr;
    }
    

    template<typename TRangeFunction>
    HDINLINE
    void
    set(TContainer*, TIndex & idx,TRange const &, TRangeFunction&)
    const
    {
        idx = nullptr;
    }
};
}
    
} // namespace traits

}// namespace hzdr

