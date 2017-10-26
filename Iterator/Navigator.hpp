
/**
 * \struct Navigator
 @brief This is the default implementation of the navigator. This navigator has
 only functionality for forward iteration. Your implementation has two special 
 parameters: The offset and the jumpsize. This parameters are used to implement 
 a parallel iterator. The offset is the distance from the beginning of the 
 container to the first element of the iterator, i.e. the process id. 
 The jumpsize is the distance between two iterator positions, i.e. the number of
 processes. All traits are instantiated as class members. 
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


namespace hzdr 
{
namespace details
{
template <typename T>
class UndefinedLastElement
{
    typedef char one;
    typedef long two;

    template <typename C> static one test( typeof(&C::UNDEFINED) ) ;
    template <typename C> static two test(...);    

public:
    enum { value = sizeof(test<T>(0)) == sizeof(char) };
}; // class UndefinedAhead

template<typename T>    
struct OffsetRangeType
{
    T v;
    typedef decltype(v()) type;
};

} // namespace details

template<
    typename TContainer,
    typename TComponent,
    typename TOffset,
    typename TJumpsize,
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
struct Navigator
{
// define the types 
    typedef typename std::decay<TContainer>::type                   ContainerType;
    typedef ContainerType*                                          ContainerPtr;
    typedef ContainerType&                                          ContainerRef;
    typedef TComponent                                              ComponentType;
    typedef ComponentType*                                          ComponentPtr;
    typedef TJumpsize                                               JumpsizeType;
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
            JumpsizeType && jumpsize):
        offset(hzdr::forward<OffsetType>(offset)),
        jumpsize(hzdr::forward<JumpsizeType>(jumpsize))
    {}
    
    
    /**
     * @brief The function moves the iterator forward to the next element. 
     * @param index in: current position of iterator; out: position of the 
     * iterator after the move.
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
        // test if the iterator is before the first element
        if(beforeFirstElement.test(containerPtr, index, offset(), containerSize))
        {
            firstElement(containerPtr, index);
            nextElement(
                containerPtr, 
                index,  
                static_cast<RangeType>(offset()),
                containerSize);
            --distance;
        }
        // We jump over distance * jumpsize elements
        auto remainingJumpsize = nextElement(
            containerPtr, 
            index,  
            static_cast<RangeType>(jumpsize() * distance),
            containerSize);
        
        /// @todo Test ob es aufgerundet werden muss
        // we need the distance from the last element to the current index position
        return static_cast<RangeType>(remainingJumpsize + jumpsize() - 1) / static_cast<RangeType>(jumpsize());
    }
    
    
    /**
     * @brief The function moves the iterator backward to the next element. 
     * @param index in: current position of iterator; out: position of the 
     * iterator after the move.
     * @result the distance from the end element to the hypothetical position
     * given by the distance parameter
     */
    HDINLINE
    RangeType
    previous(
        ContainerPtr containerPtr,  
        IndexType & index,
        RangeType distance)
    {

        // test if the iterator is after the last element
        if(afterLastElement.test(containerPtr, index, containerSize))
        {
            // set the index to the last element
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
            --distance;
        }
        // We jump over distance * jumpsize elements
        auto remainingJumpsize = previousElement(
            containerPtr, 
            index,
            offset(),
            static_cast<RangeType>(jumpsize() * distance),
            containerSize);
        

        // we need the distance from the last element to the current index position
        return static_cast<RangeType>(remainingJumpsize + jumpsize() - 1) / static_cast<RangeType>(jumpsize());
    }
    
    /**
     * @brief set the iterator to the first element
     * 
     */

    HDINLINE 
    void 
    begin(
        ContainerPtr containerPtr,  
        IndexType & index)
    {
        firstElement(containerPtr, index);
        nextElement(
            containerPtr, 
            index,  
            static_cast<RangeType>(offset()),
            containerSize);
    }
    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     */

    HDINLINE 
    void 
    rbegin(
        ContainerPtr containerPtr,  
        IndexType & index)
    {
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

        
    }
    
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
     */

    HDINLINE 
    void 
    rend(
        ContainerPtr containerPtr,  
        IndexType & index)
    {
        beforeFirstElement.set(containerPtr, index, offset(), containerSize);
    }
    
    HDINLINE 
    bool
    isAfterLast(
        ContainerPtr containerPtr,  
        IndexType const & index)
    const
    {
//         std::cout << "IsAfterLast: " << std::boolalpha <<  afterLastElement.test(containerPtr, index, containerSize) << "index " << index << " container " << containerSize(containerPtr) << std::endl;
        return afterLastElement.test(containerPtr, index, containerSize);
    }
    
    HDINLINE 
    bool
    isBeforeFirst(
        ContainerPtr containerPtr,   
        IndexType const & index)
    const
    {
        return beforeFirstElement.test(containerPtr, index, offset(), containerSize);
    }
    
    
    HDINLINE
    RangeType 
    nbElements(ContainerPtr containerPtr)
    const
    {
        return containerSize(containerPtr);
    }
    
    HDINLINE
    RangeType
    size(ContainerPtr containerPtr)
    const 
    {
        return (nbElements(containerPtr) - offset() + jumpsize() - static_cast<RangeType>(1)) / jumpsize();
    }
    
//variables
protected:
    OffsetType offset;
    JumpsizeType jumpsize;
    NumberElements containerSize;
    FirstElement firstElement;
    NextElement nextElement;
    AfterLastElement afterLastElement;
    LastElement lastElement;
    PreviousElement previousElement;
    BeforeFirstElement beforeFirstElement;
};


/**
 * @brief This navigator is a concept. It has an offset and a jumpsize.
 */

template<
    typename TOffset,
    typename TJumpsize>
struct Navigator<
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    TOffset,
    TJumpsize,
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
        typename TOffset_,
        typename TJumpsize_>
    HDINLINE
    Navigator(
            TOffset_ && offset, 
            TJumpsize_ && jumpsize):
        offset(hzdr::forward<TOffset_>(offset)),
        jumpsize(hzdr::forward<TJumpsize_>(jumpsize))
    {}
    
    OffsetType offset;
    JumpsizeType jumpsize;
};


/**
 * @brief creates an navigator concept. It needs an offset and the jumpsize
 * @param offset distance from the begining of the container to the first position
 * of the iterator 
 * @param jumpsize distance between two elements within the container
 * 
 */
template<
    typename TOffset,
    typename TJumpsize>
HDINLINE
auto 
makeNavigator(
    TOffset && offset,
    TJumpsize && jumpsize)
-> 
    hzdr::Navigator<
        details::UndefinedType,
        details::UndefinedType,
        typename std::decay<TOffset>::type,
        typename std::decay<TJumpsize>::type,
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
    typedef typename std::decay<TOffset>::type OffsetType;
    typedef typename std::decay<TJumpsize>::type JumpsizeType;
    typedef hzdr::Navigator<
        details::UndefinedType,
        details::UndefinedType,
        OffsetType,
        JumpsizeType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        false> ResultType;
    
    return ResultType(
        hzdr::forward<TOffset>(offset),
        hzdr::forward<TJumpsize>(jumpsize));
}



namespace details
{



    template<typename T>
    struct NavigatorTemplates
    {
        typedef typename std::decay<T>::type            _T;
        typedef typename _T::ContainerType              ContainerType;
        typedef typename _T::OffsetType                 OffsetType;
        typedef typename _T::JumpsizeType               JumpsizeType;
        typedef typename _T::IndexType                  IndexType;
        typedef typename _T::RangeType                  RangeType;
        typedef typename _T::NumberElements          NumberElements;
        typedef typename _T::FirstElement               FirstElement;
        typedef typename _T::NextElement                NextElement;
        typedef typename _T::AfterLastElement           AfterLastElement;
        typedef typename _T::LastElement                LastElement;
        typedef typename _T::PreviousElement            PreviousElement;
        typedef typename _T::BeforeFirstElement         BeforeFirstElement;
        
    };


template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TNavigator,
    typename TOffset = typename details::NavigatorTemplates<TNavigator>::OffsetType,
    typename TJumpsize = typename details::NavigatorTemplates<TNavigator>::JumpsizeType,
    typename TComponent = typename hzdr::traits::ComponentType<TContainerNoRef>::type,
    typename TContainerCategorie = typename hzdr::traits::ContainerCategory<TContainerNoRef>::type,
    typename TContainerSize = typename hzdr::traits::NumberElements<TContainerNoRef>,
    typename TIndex = typename hzdr::traits::IndexType<TContainerNoRef>::type,
    typename TRange = typename std::decay<typename OffsetRangeType<TOffset>::type>::type,
    typename TFirstElement = typename hzdr::traits::navigator::FirstElement<TContainerNoRef, TIndex, TContainerCategorie>,
    typename TAfterLastElement = typename hzdr::traits::navigator::AfterLastElement<TContainerNoRef, TIndex, TContainerCategorie>,
    typename TNextElement = typename hzdr::traits::navigator::NextElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>,
    typename TLastElement = typename hzdr::traits::navigator::LastElement<TContainerNoRef, TIndex, TContainerCategorie>,
    typename TPreviousElement = typename hzdr::traits::navigator::PreviousElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>,
    typename TBeforeFirstElement = typename hzdr::traits::navigator::BeforeFirstElement<TContainerNoRef, TIndex, TRange, TContainerCategorie>,
    bool isBidirectional = not std::is_same<TLastElement, hzdr::details::UndefinedType>::value>
auto
HDINLINE
makeNavigator(
    TNavigator && navi)
->
hzdr::Navigator<
    TContainerNoRef,
    TComponent,
    TOffset,
    TJumpsize,
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
    typedef hzdr::Navigator<
        TContainerNoRef,
        TComponent,
        TOffset,
        TJumpsize,
        TIndex,
        TContainerSize,
        TRange,
        TFirstElement,
        TNextElement,
        TAfterLastElement,
        TLastElement,
        TPreviousElement,
        TBeforeFirstElement,
        isBidirectional> ResultType;
    return ResultType(hzdr::forward<TOffset>(navi.offset), hzdr::forward<TJumpsize>(navi.jumpsize));
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
    TJumpsize && jumpsize)
-> 
    hzdr::Navigator<
        TContainerNoRef,
        TComponent,
        TOffset,
        TJumpsize,
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

    typedef hzdr::Navigator<
        TContainerNoRef,
        TComponent,
        TOffset,
        TJumpsize,
        TIndex,
        TContainerSize,
        TRange,
        TFirstElement,
        TNextElement,
        TAfterLastElement,
        TLastElement,
        TPreviousElement,
        TBeforeFirstElement,
        isBidirectional> ResultType;
    
    return ResultType(
        hzdr::forward<TOffset>(offset),
        hzdr::forward<TJumpsize>(jumpsize));
}

}// namespace hzdr

