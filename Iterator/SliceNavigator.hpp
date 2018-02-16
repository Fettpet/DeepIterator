/**
 * \struct Navigator
 @brief This is the default implementation of the navigator. The task of the 
 navigator is to define the first element, the next element and an after last 
 element. If the navigator is bidirectional it need also a last element, a previous
 element and a before first element. 
 
 The navigator has two traits for parallel 
 walking through the container. The first one is TOffset. This is used to get the 
 distance from the first element of the container to the first element which will
 be accessed. This trait can be used to map the thread ID (for example offset 
 = threadIdx.x). The second trait is the jumpsize. The jumpsize is the distance
 between two iterator elements. The number of threads can be mapped on the jump-
 size. With this two traits you can go parallel over all elements and touch each
 element only one times. 
 
 We had three/six traits for the behaviour of the container. The first three traits
 are 
 1. define the first element of the container,
 2. define a next element of the container,
 3. define a after last element of the container.
 If the navigator is bidirectional three additional traits are needed
 4. define the last element within the container
 5. define a previous element of the container
 6. define a before first element of the container.
 The navigator use this 8 traits to define methodes for parallel iteration though
 the container.
 
 @tparam TContainer Type of the container,
 @tparam TComponent Type of the component of the container.
 @tparam TOffset Policy to get the offset. You need to specify the () operator.
 @tparam TJumpsize Policy to specify the Jumpsize. It need the operator ().
 @tparam TIndex Type of the index. The index is used to specify the iterator 
 position.
 @tparam TContainerSize Trait to specify the size of a container. It need the 
 function operator()(TContainer*). TContainer is a pointer to the container 
 instance over which the iterator walks.
 @tparam TFirstElement Trait to set the index to the first element. It need the 
 function operator()(TContainer*, TIndex&, const TRange). TRange is the result 
 type of TOffset's (). TContainer is a pointer to the container 
 instance over which the iterator walks. TIndex is used to describe the position.
 TRange is the offset.
 @tparam TNextElement Trait to set the index to the next element. The trait need 
 the function TRange operator()(TContainer*, TIndex&, TRange). The TRange 
 parameter is used to handle the jumpsize. The result of this function is the 
 remaining jumpsize. A little example. Your container has 10 elements and your
 iterator is the the 8 element. Your jumpsize is 5. This means the new position
 would be 13. So the result of the function is 3, the remaining jumpsize.
 @tparam TAfterLastElement This Trait is used to check whether the iteration is 
 after the last element. The function header is 
 bool operator()(TContainer*, TIndex&). It returns true, if the end is reached, 
 and false otherwise.
 @tparam TLastElement This trait gives the last element which the iterator would
 access, befor the end is reached, in a forward iteration case. The function 
 head is operator()(TContainer*, TIndex&, const TRange). This trait is only needed
 if the navigator is bidirectional. 
 @tparam TPreviousElement Trait to set the index to the previous element. The 
 trait need the function TRange operator()(TContainer*, TIndex&, TRange). This 
 trait is only needed if the navigator is bidirectional. For fourther 
 informations see TNextElement.
 @tparam TBeforeFirstElement Used to check whether the iterator is before the
 first element. The function header is bool operator()(TContainer*, TIndex&). 
 It returns true, if the end is reached, and false otherwise.
 @tparam isBidirectional Set the navigator to bidirectional (true) or to forward
 only (false)
 */

#pragma once
#include "Definitions/forward.hpp"
#include "Policies.hpp"
#include "PIC/Frame.hpp"
#include <boost/core/ignore_unused.hpp>
#include "PIC/Supercell.hpp"
#include <type_traits>
#include "Definitions/hdinline.hpp"
#include "Traits/Componenttype.hpp"
#include "Traits/ContainerCategory.hpp"
#include "Traits/RangeType.hpp"
#include "Traits/Navigator/AfterLastElement.hpp"
#include "Traits/Navigator/BeforeFirstElement.hpp"
#include "Traits/Navigator/LastElement.hpp"
#include "Traits/Navigator/NextElement.hpp"
#include "Traits/Navigator/PreviousElement.hpp"
#include "Traits/Navigator/FirstElement.hpp"
#include "Iterator/Categorie.hpp"
#include <cassert>

namespace hzdr 
{
namespace slice
{
    struct NumberElements;
    struct IgnoreLastElements;
}
    
template<
    typename _start_begin,
    int_fast32_t _distance
>
struct Slice;

template<
    int_fast32_t _distance
>
struct Slice<
    hzdr::slice::NumberElements,
    _distance
>
{
public:
    
    constexpr
    auto 
    distance()
    const
    ->
    int_fast32_t
    {
        return _distance;
    }
    
    constexpr 
    auto 
    from_start()
    const
    ->
    bool
    {
        return true;
    }
    
};


template<
    int_fast32_t _distance
>
struct Slice<
    hzdr::slice::IgnoreLastElements,
    _distance
>
{
public:
    
    constexpr
    auto 
    distance()
    const
    ->
    int_fast32_t
    {
        return _distance;
    }
    
    constexpr 
    auto 
    from_start()
    const
    ->
    bool
    {
        return false;
    }
    
};

template<
    typename TContainer,
    typename TComponent,
    typename TOffset,
    typename TJumpsize,
    typename TSlice,
    typename TIndex,
    typename TContainerSize,
    typename TRange,
    typename TFirstElement,
    typename TNextElement,
    typename TAfterLastElement,
    typename TLastElement = hzdr::details::UndefinedType,
    typename TPreviousElement = hzdr::details::UndefinedType,
    typename TBeforeFirstElement = hzdr::details::UndefinedType,
    bool isBidirectional = not details::UndefinedLastElement<TLastElement>::value
>
struct SlicedNavigator
{
// define the types 
    typedef typename std::decay<TContainer>::type                   ContainerType;
    typedef ContainerType*                                          ContainerPtr;
    typedef ContainerType&                                          ContainerRef;
    typedef TComponent                                              ComponentType;
    typedef ComponentType*                                          ComponentPtr;
    typedef TJumpsize                                               JumpsizeType;
    typedef TSlice                                                  SliceType;
    typedef TOffset                                                 OffsetType;
    typedef TIndex                                                  IndexType;
    typedef TRange                                                  RangeType;
    typedef TContainerSize                                          NumberElements;
    typedef TFirstElement                                           FirstElement;
    typedef TNextElement                                            NextElement;
    typedef TAfterLastElement                                       AfterLastElement;
    typedef TLastElement                                            LastElement;
    typedef TPreviousElement                                        PreviousElement;
    typedef TBeforeFirstElement                                     BeforeFirstElement;

    
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
        cur_pos(0),
        offset(hzdr::forward<OffsetType>(offset)),
        jumpsize(hzdr::forward<JumpsizeType>(jumpsize)),
        slice(hzdr::forward<SliceType>(slice)),
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
    RangeType
    next(
        ContainerPtr containerPtr,  
        IndexType & index,
        RangeType distance)
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        // We jump over distance * jumpsize elements
        auto remainingJumpsize = nextElement(
            containerPtr, 
            index,  
            static_cast<RangeType>(jumpsize() * distance),
            containerSize
        );
        
        cur_pos += distance;
        auto nbElem = size(containerPtr);
        // we need the distance from the last element to the current index position
        // this is a round up
       
        // 1. We start counting from the begininng and the position is outside
        // the slice.
        if( slice.from_start() && (cur_pos > slice.distance()))
        {
            return cur_pos - slice.distance();
        }
        // 2. We ignore the last elements
        else if(not slice.from_start() && (cur_pos > static_cast<int_fast32_t>(nbElem - slice.distance())))
        {
            return cur_pos + slice.distance() - nbElem;
        }
        // 3. if it is outside the container 
        else 
        {
            return  static_cast<RangeType>(remainingJumpsize + jumpsize() - 1) / static_cast<RangeType>(jumpsize());
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
    typename std::enable_if<T==true, RangeType>::type
    previous(
        ContainerPtr containerPtr,  
        IndexType & index,
        RangeType distance)
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        // We jump over distance * jumpsize elements
        auto remainingJumpsize = previousElement(
            containerPtr, 
            index,
            offset(),
            static_cast<RangeType>(jumpsize() * distance),
            containerSize
        );
        
        cur_pos -= distance;
        auto nbElem = size(containerPtr);
        // we need the distance from the last element to the current index position
        // this is a round up
       
        // 1. We start counting from the begininng and the position is outside
        // the slice.
        if( slice.from_start() && (-1 * cur_pos > slice.distance()))
        {
            return -1 * cur_pos - slice.distance();
        }
        // 2. We ignore the last elements
        // The cast is nessacary since the container could be empty
        else if(not slice.from_start() && (static_cast<int_fast32_t>(-1 * cur_pos) > static_cast<int_fast32_t>(nbElem - slice.distance())))
        {
            return -1 * cur_pos + slice.distance() - nbElem;
        }
        // 3. if it is outside the container 
        else 
        {
            return  static_cast<RangeType>(remainingJumpsize + jumpsize() - 1) / static_cast<RangeType>(jumpsize());
        }
    }
    
    /**
     * @brief set the iterator to the first element
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: first element of the iterator.
     */
    HDINLINE 
    void 
    begin(
        ContainerPtr containerPtr,  
        IndexType & index)
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        firstElement(containerPtr, index);
        nextElement(
            containerPtr, 
            index,  
            static_cast<RangeType>(offset()),
            containerSize);
        cur_pos = 0;
    }
    
    /**
     * @brief set the iterator to the last element. 
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: last element of the iterator.
     */

    template< bool T=isBidirectional>
    HDINLINE 
    typename std::enable_if<T==true>::type
    rbegin(
        ContainerPtr containerPtr,  
        IndexType & index)
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        auto nbElementsVar = nbElements(containerPtr);
        // -1 since we dont like to jump outside
        auto nbJumps = (nbElementsVar - offset() - 1) / jumpsize();
        auto lastPosition = nbJumps * jumpsize() + offset();
        // -1 since we need the last position
        auto neededJumps = (nbElementsVar - 1) - lastPosition;

        lastElement(containerPtr, index, containerSize);
        previousElement(
            containerPtr, 
            index,
            offset(),
            static_cast<RangeType>(neededJumps),
            containerSize);
        cur_pos = 0;
        
    }
    
    /**
     * @brief set the iterator to the after last element
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: index of the after last element
     */
    HDINLINE 
    void 
    end(
        ContainerPtr containerPtr,  
        IndexType & index)
    {
        afterLastElement.set(containerPtr, index, containerSize);
    }
    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     * @param containerPtr pointer to the container, over which we iterate
     * @param index out: index of the before first element
     */

    template< bool T=isBidirectional>
    HDINLINE 
    typename std::enable_if<T==true>::type
    rend(
        ContainerPtr containerPtr,  
        IndexType & index)
    {
        beforeFirstElement.set(
            containerPtr, 
            index, 
            offset(), 
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
        IndexType const & index)
    const
    {
        
        /*there are three cases: 
         * 1. if the trait say it after the last element
         * 2. if slice.from_start() and cur_pos > slice.distance()
         * 3. not slice.from_start() and cur_pos > nbElements - offset - slice.distance()
         */
        return 
             afterLastElement.test(containerPtr, index, containerSize)
          || (slice.from_start() && (cur_pos >  ( slice.distance())))
          || (not slice.from_start() && (cur_pos * jumpsize() + offset() >= 
                nbElements(containerPtr) - slice.distance() ));
    }
    
    /**
     * @brief check wheter the index is before the first element
     * @param containerPtr pointer to the container, over which we iterate
     * @param index in: current index position
     * @return true, if the index is before the first element; false otherwise
     */
    template< bool T=isBidirectional>
    HDINLINE 
    typename std::enable_if<T==true, bool>::type
    isBeforeFirst(
        ContainerPtr containerPtr,   
        IndexType const & index)
    const
    {
        return 
            beforeFirstElement.test(containerPtr, index, offset(), containerSize)
            || (slice.from_start() && (cur_pos < -1 * slice.distance()))
            || (not slice.from_start() && (static_cast<int_fast32_t>(cur_pos * jumpsize() - offset()) <= 
                -1 * static_cast<int_fast32_t>(nbElements(containerPtr) - slice.distance() )));
    }
    
    /**
     * @brief this function determine the number of elements within the container
     * @param containerPtr pointer to the container, you like to know the number
     * of elements
     * @return number of elements within the container
     */
    HDINLINE
    RangeType 
    nbElements(ContainerPtr containerPtr)
    const
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
    RangeType
    size(ContainerPtr containerPtr)
    const 
    {
        assert(containerPtr != nullptr); // containerptr should be valid
        
        auto const nbElem = nbElements(containerPtr);
        auto const off = offset();
        
        if(slice.from_start())
        {
            

            
            // 1. Case nbElem - off > slice.distance()
            uint sizeFirstCase = (
                    slice.distance()
                    + jumpsize() )
                / jumpsize();
            // 2. Case nbElem - off < slice.distance()
            uint sizeSecondCase = (
                    nbElem 
                    - off 
                    + jumpsize() 
                    - static_cast<RangeType>(1)) 
                / jumpsize();

            // check and give it back
            return ((static_cast<int_fast32_t>(nbElem - off) >= slice.distance()) * sizeFirstCase 
                + (static_cast<int_fast32_t>(nbElem) < slice.distance()) * sizeSecondCase);
        }
        // it ignores the last slice.distance() elements
        else 
        {
            // 1. Case nbElem - off > slice.distance()
            // I had nbElem - off - slice.distance() elements
                uint sizeFirstCase = (
                    nbElem - off - slice.distance()
                    + jumpsize() 
                    - static_cast<RangeType>(1))
                / jumpsize();
            // 2. Case nbElem - off < slice.distance()
            // I had 0 elements inside
            
                
            return (off < nbElem) * (static_cast<int_fast32_t>(nbElem - off) > slice.distance()) * sizeFirstCase;
                
        }
    }
    
//variables
protected:
    int cur_pos;
    OffsetType offset;
    JumpsizeType jumpsize;
    SliceType slice;
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
    typename TOffset,
    typename TJumpsize,
    typename TSlice>
struct SlicedNavigator<
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    TOffset,
    TJumpsize,
    TSlice,
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
    typedef hzdr::details::UndefinedType                            ContainerType;
    typedef ContainerType*                                          ContainerPtr;
    typedef ContainerType&                                          ContainerRef;
    typedef hzdr::details::UndefinedType                            ComponentType;
    typedef ComponentType*                                          ComponentPtr;
    typedef TJumpsize                                               JumpsizeType;
    typedef TOffset                                                 OffsetType;
    typedef TSlice                                                  SliceType;
    typedef hzdr::details::UndefinedType                            IndexType;
    typedef hzdr::details::UndefinedType                            RangeType;
    typedef hzdr::details::UndefinedType                            NumberElements;
    typedef hzdr::details::UndefinedType                            FirstElement;
    typedef hzdr::details::UndefinedType                            NextElement;
    typedef hzdr::details::UndefinedType                            AfterLastElement;
    typedef hzdr::details::UndefinedType                            LastElement;
    typedef hzdr::details::UndefinedType                            PreviousElement;
    typedef hzdr::details::UndefinedType                            BeforeFirstElement;
    
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
        typename TOffset_,
        typename TJumpsize_,
        typename TSlice_>
    HDINLINE
    SlicedNavigator(
            TOffset_ && offset, 
            TJumpsize_ && jumpsize,
            TSlice_ && slice
             ):
        offset(hzdr::forward<TOffset_>(offset)),
        jumpsize(hzdr::forward<TJumpsize_>(jumpsize)),
        slice(hzdr::forward<TSlice_>(slice))
    {}
    
    OffsetType offset;
    JumpsizeType jumpsize;
    SliceType slice;
    
} ;


/**
 * @brief creates an navigator concept. It needs an offset and the jumpsize
 * @param offset distance from the begining of the container to the first position
 * of the iterator 
 * @param jumpsize distance between two elements within the container
 * 
 */
template<
    typename TOffset,
    typename TJumpsize,
    typename TSlice>
HDINLINE
auto 
makeNavigator(
    TOffset && offset,
    TJumpsize && jumpsize,
    TSlice && slice
             )
-> 
    hzdr::SlicedNavigator<
        details::UndefinedType,
        details::UndefinedType,
        typename std::decay<TOffset>::type,
        typename std::decay<TJumpsize>::type,
        typename std::decay<TSlice>::type,
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
    using OffsetType = typename std::decay<TOffset>::type ;
    using JumpsizeType = typename std::decay<TJumpsize>::type ;
    using SliceType = typename std::decay<TSlice>::type;
    using ResultType =  hzdr::SlicedNavigator<
        details::UndefinedType,
        details::UndefinedType,
        OffsetType,
        JumpsizeType,
        SliceType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        false>;
    
    auto && result = ResultType(
        hzdr::forward<TOffset>(offset),
        hzdr::forward<TJumpsize>(jumpsize),
        hzdr::forward<TSlice>(slice)
    );
    return result;
}



namespace details
{

    template<
        typename T,
        typename _T = typename std::decay<T>::type,
        typename TSliceType = typename _T::SliceType,
        typename TContainerType = typename _T::ContainerType,
        typename TOffset = typename _T::OffsetType,
        typename TJumpsize = typename _T::JumpsizeType,
        typename TIndex = typename _T::IndexType,
        typename TRange = typename _T::RangeType,
        typename TNumberElements = typename _T::NumberElements,
        typename TFirstElement = typename _T::FirstElement,
        typename TNextElement = typename _T::NextElement,
        typename TAfterLastElement = typename _T::AfterLastElement,
        typename TLast = typename _T::LastElement,
        typename TPrevious = typename _T::PreviousElement,
        typename TBeforeFirst = typename _T::BeforeFirstElement
    >
    struct SlicedNavigatorTemplates
    {
        using ContainerType = TContainerType;
        using OffsetType = TContainerType;
        using JumpsizeType = TJumpsize;
        using SliceType = TSliceType;
        using IndexType = TIndex;
        using RangeType = TRange;
        using NumberElements = TNumberElements;
        using FirstElement = TFirstElement;
        using NextElement = TNextElement;
        using AfterLastElement = TAfterLastElement;
        using LastElement = TLast;
        using PreviousElement = TPrevious;
        using BeforeFirstElement = TBeforeFirst;
    };


template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename OffsetType,
    typename JumpsizeType,
    typename SliceType,
    typename TComponent = typename hzdr::traits::ComponentType<TContainerNoRef>::type,
    typename TContainerCategorie = typename hzdr::traits::ContainerCategory<TContainerNoRef>::type,
    typename TContainerSize = typename hzdr::traits::NumberElements<TContainerNoRef>,
    typename TIndex = typename hzdr::traits::IndexType<TContainerNoRef>::type,
    typename TRange = typename std::decay<typename OffsetRangeType<typename std::decay<OffsetType>::type >::type>::type,
    typename TFirstElement = typename hzdr::traits::navigator::FirstElement<TContainerNoRef, TIndex, TContainerCategorie>,
    typename TAfterLastElement = typename hzdr::traits::navigator::AfterLastElement<TContainerNoRef, TIndex, TContainerCategorie>,
    typename TNextElement = typename hzdr::traits::navigator::NextElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>,
    typename TLastElement = typename hzdr::traits::navigator::LastElement<TContainerNoRef, TIndex, TContainerCategorie>,
    typename TPreviousElement = typename hzdr::traits::navigator::PreviousElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>,
    typename TBeforeFirstElement = typename hzdr::traits::navigator::BeforeFirstElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>,
    bool isBidirectional = not std::is_same<TLastElement, hzdr::details::UndefinedType>::value
>
auto
HDINLINE
makeNavigator(
    hzdr::SlicedNavigator<
        details::UndefinedType,
        details::UndefinedType,
        OffsetType,
        JumpsizeType,
        SliceType,
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
    > & navi
)
->
hzdr::SlicedNavigator<
    TContainerNoRef,
    TComponent,
    OffsetType,
    JumpsizeType,
    SliceType,
    TIndex,
    TContainerSize,
    TRange,
    TFirstElement,
    TNextElement,
    TAfterLastElement,
    TLastElement,
    TPreviousElement,
    TBeforeFirstElement,
    isBidirectional>
{
    using ResultType = hzdr::SlicedNavigator<
        TContainerNoRef,
        TComponent,
        OffsetType,
        JumpsizeType,
        SliceType,
        TIndex,
        TContainerSize,
        TRange,
        TFirstElement,
        TNextElement,
        TAfterLastElement,
        TLastElement,
        TPreviousElement,
        TBeforeFirstElement,
        isBidirectional>;
        

    auto && result = ResultType(
        hzdr::forward<OffsetType>(navi.offset), 
        hzdr::forward<JumpsizeType>(navi.jumpsize),
        hzdr::forward<SliceType>(navi.slice)
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
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TOffset,
    typename TJumpsize,
    typename TSlice,
    typename TComponent = typename hzdr::traits::ComponentType<TContainerNoRef>::type,
    typename TContainerCategorie = typename hzdr::traits::ContainerCategory<TContainerNoRef>::type,
    typename TContainerSize = typename hzdr::traits::NumberElements<TContainerNoRef>::type,
    typename TIndex = typename hzdr::traits::IndexType<TContainerNoRef>::type,
    typename TRange = decltype(TOffset::operator()()),
    typename TFirstElement = typename hzdr::traits::navigator::FirstElement<TContainerNoRef, TIndex, TContainerCategorie>::type,
    typename TAfterLastElement = typename hzdr::traits::navigator::AfterLastElement<TContainerNoRef, TIndex, TContainerCategorie>::type,
    typename TNextElement = typename hzdr::traits::navigator::NextElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>::type,
    typename TLastElement = typename hzdr::traits::navigator::LastElement<TContainerNoRef, TIndex, TContainerCategorie>::type,
    typename TPreviousElement = typename hzdr::traits::navigator::PreviousElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>::type,
    typename TBeforeFirstElement = typename hzdr::traits::navigator::BeforeFirstElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>::type,
    bool isBidirectional = not std::is_same<TLastElement, hzdr::details::UndefinedType>::value>
auto 
HDINLINE
makeNavigator(
    TOffset && offset,
    TJumpsize && jumpsize,
    TSlice && slice
             )
-> 
    hzdr::SlicedNavigator<
        TContainerNoRef,
        TComponent,
        typename std::decay<TOffset>::type,
        typename std::decay<TJumpsize>::type,
        typename std::decay<TSlice>::type,
        TIndex,
        TContainerSize,
        TRange,
        TFirstElement,
        TNextElement,
        TAfterLastElement,
        TLastElement,
        TPreviousElement,
        TBeforeFirstElement,
        isBidirectional>
{

    using ResultType = hzdr::SlicedNavigator<
        TContainerNoRef,
        TComponent,
        typename std::decay<TOffset>::type,
        typename std::decay<TJumpsize>::type,
        typename std::decay<TSlice>::type,
        TIndex,
        TContainerSize,
        TRange,
        TFirstElement,
        TNextElement,
        TAfterLastElement,
        TLastElement,
        TPreviousElement,
        TBeforeFirstElement,
        isBidirectional>;
        
    auto && result = ResultType(
        hzdr::forward<TOffset>(offset),
        hzdr::forward<TJumpsize>(jumpsize),
        hzdr::forward<TSlice>(slice)
    );
    
    return result;
}

}// namespace hzdr

