
/**
 * \struct Navigator
 * @author Sebastian Hahn (t.hahn< at >hzdr.de )
 * ove
 * @brief The navigator is used to move through a container. 
 * 
 * @tparam TContainer The datatype of the datastructure. If the datastructure is
 * indexable you doesnt need to write your own navigator. 
 * It has three templates:
 
 
 */

#pragma once
#include "Policies.hpp"
#include "PIC/Frame.hpp"
#include <boost/core/ignore_unused.hpp>
#include "PIC/Supercell.hpp"
#include <type_traits>
#include "Definitions/hdinline.hpp"
#include "Traits/Componenttype.hpp"
#include "Traits/ContainerCategory.hpp"
#include "Traits/RangeType.hpp"
#include "Iterator/Categorie/DoublyLinkListLike.hpp"

namespace hzdr 
{


/**
 * @brief This is the default implementation of the navigator. This navigator has
 * only functionality for forward iteration. To use this one you need to specify 
 * the following traits:
 * 1. ComponentType
 * 2. IndexType
 * 3. NumberElements
 * 4. container::categorie::NextElement
 * 5. container::categorie::EndElementReached
 * 6. container::categorie::FirstElement
 * 
 * @tparam TContainer Type of the container,
 * @tparam TOffset policy to get the offset. You need to specify the () operator.

 */
template<
    typename TContainer,
    typename TOffset,
    typename TJumpsize,
    typename TContainerCategorie = void,
    typename SFINAE = void>
struct Navigator
{
// define the types 
    typedef typename std::decay<TContainer>::type                   ContainerType;
    typedef ContainerType*                                          ContainerPtr;
    typedef ContainerType&                                          ContainerRef;
    typedef typename hzdr::traits::ComponentType<ContainerType>::type ComponentType;
    typedef ComponentType*                                           ComponentPtr;
    typedef TJumpsize                                               JumpsizeType;
    typedef TOffset                                                 OffsetType;
    typedef hzdr::container::categorie::DoublyLinkListLike          ContainerCategoryType;
    typedef typename traits::IndexType<ContainerCategoryType>::type IndexType;
    typedef typename traits::RangeType<ContainerCategoryType>::type RangeType;
    typedef traits::NumberElements<ContainerType>                   ContainerSizeType;
    
// the default constructors
    HDINLINE Navigator() = default;
    HDINLINE Navigator(Navigator const &) = default;
    HDINLINE Navigator(Navigator &&) = default;
    HDINLINE ~Navigator() = default;

    
public:
    /**
     * @brief Set the offset and the jumpsize to the given values
       @param offset the distance from the start to the first element
       @param jumpsize distance between two elements
    */
    HDINLINE
    Navigator(
            OffsetType && offset, 
            JumpsizeType && jumpsize):
        offset(std::forward<OffsetType>(offset)),
        jumpsize(std::forward<JumpsizeType>(jumpsize))
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
        ContainerPtr,  
        ComponentPtr& componentPtr,
        IndexType & index,
        RangeType const & distance)
    {

        container::categorie::NextElement<ContainerType> nextElement;
        container::categorie::EndElementReached<ContainerType> endReached;

        RangeType counter = 0;
        for(; counter<distance; ++counter)
        {
            for(decltype(jumpsize()) i=0; i< jumpsize(); ++i)
            {
                componentPtr = nextElement(componentPtr);
                if(endReached(componentPtr))
                {
                    index = 1;
                    break;
                    
                }
            }
            if(endReached(componentPtr))
                    break;
        }
        // we need the distance from the last element to the current index position
        return distance - counter;
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
        ContainerPtr,  
        ComponentPtr& componentPtr,
        IndexType & index,
        RangeType distance)
    {
        container::categorie::PreviousElement<ContainerType> previousElement;
        container::categorie::NextElement<ContainerType> nextElement;
        container::categorie::BeforeElementReached<ContainerType> beforeReached;
        
        RangeType counter = 0;
        for(; counter<distance; ++counter)
        {
            for(decltype(jumpsize()) i=static_cast<decltype(jumpsize())>(0); i< jumpsize(); ++i)
            {
                componentPtr = previousElement(componentPtr);
                if(beforeReached(componentPtr))
                {
                    break;
                    index = -1;
                }
            }
            if(beforeReached(componentPtr))
                     return distance - counter;
        }
        std::cout << "works" << std::endl;
        // check whether the element is before the offset element
        for(auto i=0u; i < offset(); ++i)
        {
            
            componentPtr = previousElement(componentPtr);
            if(beforeReached(componentPtr))
            {
                
                return 0;
            }
        }
        
        for(auto i=0u; i < offset(); ++i)
        {
            componentPtr = nextElement(componentPtr);
        }
        
        // we need the distance from the last element to the current index position
        return distance - counter;
    }
    
    /**
     * @brief set the iterator to the first element
     * 
     */

    HDINLINE 
    void 
    begin(
        ContainerPtr containerPtr,  
        ComponentPtr& componentPtr,
        IndexType & index)
    {
        index = 0;
        container::categorie::FirstElement<ContainerType > first;
        componentPtr = first(containerPtr);
    }
    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     */

    HDINLINE 
    void 
    rbegin(
        ContainerPtr containerPtr,  
        ComponentPtr& componentPtr,
        IndexType & index)
    {
        auto nbelem = size(containerPtr);
        auto jumps = (nbelem % jumpsize()) - ((nbelem - offset()) % jumpsize()); 
        container::categorie::BeforeElementReached<ContainerType> beforeReached;
        container::categorie::PreviousElement<ContainerType> previousElement;
        container::categorie::LastElement<ContainerType > last;
        componentPtr = last(containerPtr);
        index = static_cast<IndexType>(0);
        for( auto i=0u; i<jumps; ++i)
        {
            if(beforeReached(componentPtr))
                break;
            componentPtr = previousElement(componentPtr);
        }
    }
    
    HDINLINE 
    void 
    end(
        ContainerPtr containerPtr,  
        ComponentPtr& componentPtr,
        IndexType & index)
    {
        index = static_cast<IndexType>(1);
        container::categorie::FirstElement<ContainerType > first;
        componentPtr = first(containerPtr);
    }
    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     */

    HDINLINE 
    void 
    rend(
        ContainerPtr containerPtr,  
        ComponentPtr& componentPtr,
        IndexType & index)
    {
        container::categorie::LastElement<ContainerType > first;
        componentPtr = first(containerPtr);
        index = static_cast<IndexType>(-1);
    }
    
    HDINLINE 
    bool
    isAfterLast(
        ContainerPtr,  
        ComponentPtr comPtr,
        IndexType const & index)
    const
    {
        
        container::categorie::EndElementReached<ContainerType> endReached;
        return index == 1 or endReached(comPtr);
    }
    
    HDINLINE 
    bool
    isBeforeFirst(
        ContainerPtr,  
        ComponentPtr comPtr,
        IndexType const & index)
    const
    {
        container::categorie::BeforeElementReached<ContainerType> beforeReached;
        return index == -1 or beforeReached(comPtr);
    }
    
    uint_fast32_t size(ContainerPtr containerPtr)
    {
        uint_fast32_t counter = 0u;
        container::categorie::FirstElement<ContainerType > first;
        container::categorie::EndElementReached<ContainerType> endReached;
        container::categorie::NextElement<ContainerType> nextElement;
        auto componentPtr = first(containerPtr);
        while(not endReached(componentPtr))
        {
            ++counter;
            componentPtr = nextElement(componentPtr);
        };
        return counter;
    }
    
    HDINLINE
    int_fast32_t
    nbElements( ContainerPtr conPtr)
    const 
    {
        return (size(conPtr) - offset() + jumpsize() - 1) / jumpsize();
    }
    
//variables
protected:
    OffsetType offset;
    JumpsizeType jumpsize;
};


template<
    typename TContainer,
    typename TOffset,
    typename TJumpsize>
struct Navigator<
    TContainer,
    TOffset,
    TJumpsize,
    hzdr::container::categorie::ArrayLike>
{
// define the types 
    typedef typename std::decay<TContainer>::type                   ContainerType;
    typedef ContainerType*                                          ContainerPtr;
    typedef ContainerType&                                          ContainerRef;
    typedef typename hzdr::traits::ComponentType<ContainerType>::type ComponentType;
    typedef ComponentType*                                           ComponentPtr;
    typedef TJumpsize                                               JumpsizeType;
    typedef TOffset                                                 OffsetType;
    typedef hzdr::container::categorie::ArrayLike           ContainerCategoryType;
    typedef typename traits::IndexType<ContainerCategoryType>::type IndexType;
    typedef typename traits::RangeType<ContainerCategoryType>::type RangeType;
    typedef traits::NumberElements<ContainerType>                   ContainerSizeType;
    
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
    HDINLINE
    Navigator(
            OffsetType && offset, 
            JumpsizeType && jumpsize):
        offset(std::forward<OffsetType>(offset)),
        jumpsize(std::forward<JumpsizeType>(jumpsize))
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
        ComponentPtr,
        IndexType & index,
        RangeType const & distance)
    {
        ContainerSizeType size;
        index += distance * jumpsize();
        
        // we need the distance from the last element to the current index position
        return (index - size(containerPtr) + jumpsize() - static_cast<IndexType>(1)) / jumpsize();
    }
    
    
    /**
     * @brief The function moves the iterator backward to the next element. 
     * @param index in: current position of iterator; out: position of the 
     * iterator after the move.
     * @result the distance from the end element to the hypothetical position
     * given by the distance parameter
     */
     
    template< typename TRange_>
    HDINLINE
    RangeType
    previous(
        ContainerPtr,  
        ComponentPtr,
        IndexType & index,
        TRange_ && distance)
    {
        index -= distance * jumpsize();
        // index < 0
        // we need the distance from the last element to the current index position
        return (static_cast<IndexType>(-1) * index + jumpsize() - static_cast<IndexType>(1)) / jumpsize();
    }
    
    /**
     * @brief set the iterator to the first element
     * 
     */

    HDINLINE 
    void 
    begin(
        ContainerPtr,  
        ComponentPtr,
        IndexType & index)
    {

        index = offset();
    }
    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     */

    HDINLINE 
    void 
    rbegin(
        ContainerPtr conPtr,  
        ComponentPtr,
        IndexType & index)
    {

        ContainerSizeType size;
        auto nbOfJumps = ((size(conPtr) - offset() - 1) / jumpsize() );
        
        index = (nbOfJumps) * jumpsize() + offset();
    }
    
    
        /**
     * @brief set the iterator to the first element
     * 
     */

    HDINLINE 
    void 
    end(
        ContainerPtr conPtr,  
        ComponentPtr,
        IndexType & index)
    {
        ContainerSizeType size;
        /**
         * We need the index of the first Element outside the container
         */
        auto nbOfJumps = ((size(conPtr) - offset()-1) / jumpsize() );
        
        index = (nbOfJumps +1) * jumpsize() + offset();
    }
    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     */

    HDINLINE 
    void 
    rend(
        ContainerPtr conPtr,  
        ComponentPtr,
        IndexType & index)
    {

                /**
         * We need the index of the first Element outside the container
         */
        index = -(nbElements(conPtr) - offset()) % jumpsize() - jumpsize() - 1;
    }
    
    HDINLINE 
    bool
    isAfterLast(
        ContainerPtr conPtr,  
        ComponentPtr,
        IndexType const & index)
    const
    {
        return index >= nbElements(conPtr);
    }
    
    HDINLINE 
    bool
    isBeforeFirst(
        ContainerPtr,  
        ComponentPtr,
        IndexType const & index)
    const
    {
        return index < static_cast<IndexType>(offset());
    }
    
    HDINLINE
    int_fast32_t
    nbElements( ContainerPtr conPtr)
    const 
    {
        ContainerSizeType size;
        return size(conPtr);
    }
    
    HDINLINE
    int_fast32_t
    distanceToEnd( 
        ContainerPtr conPtr, 
        IndexType const & index)
    const 
    {
        return (nbElements(conPtr) - index + jumpsize() - 1) / jumpsize();
    }
    
    HDINLINE
    int_fast32_t
    distanceToBegin( 
        ContainerPtr, 
        IndexType const & index)
    const 
    {
        return (index - offset() + jumpsize() - 1) / jumpsize();
    }
    
    
        HDINLINE
    int_fast32_t
    size( 
        ContainerPtr conPtr)
    const 
    {
        return (nbElements(conPtr) - offset() + jumpsize() - 1) / jumpsize();
    }
    
// variables
protected:
    OffsetType offset;
    JumpsizeType jumpsize;
};



/** *************************************************************************
 * @brief The Doubly Link list like
 * ************************************************************************/


template<
    typename TContainer,
    typename TOffset,
    typename TJumpsize>
struct Navigator<
    TContainer,
    TOffset,
    TJumpsize,
    hzdr::container::categorie::DoublyLinkListLike>
{
// define the types 
    typedef typename std::decay<TContainer>::type                   ContainerType;
    typedef ContainerType*                                          ContainerPtr;
    typedef ContainerType&                                          ContainerRef;
    typedef typename hzdr::traits::ComponentType<ContainerType>::type ComponentType;
    typedef ComponentType*                                           ComponentPtr;
    typedef TJumpsize                                               JumpsizeType;
    typedef TOffset                                                 OffsetType;
    typedef hzdr::container::categorie::DoublyLinkListLike          ContainerCategoryType;
    typedef typename traits::IndexType<ContainerCategoryType>::type IndexType;
    typedef typename traits::RangeType<ContainerCategoryType>::type RangeType;
    typedef traits::NumberElements<ContainerType>                   ContainerSizeType;
    
// the default constructors
    HDINLINE Navigator() = default;
    HDINLINE Navigator(Navigator const &) = default;
    HDINLINE Navigator(Navigator &&) = default;
    HDINLINE ~Navigator() = default;

    
public:
    /**
     * @brief Set the offset and the jumpsize to the given values
       @param offset the distance from the start to the first element
       @param jumpsize distance between two elements
    */
    HDINLINE
    Navigator(
            OffsetType && offset, 
            JumpsizeType && jumpsize):
        offset(std::forward<OffsetType>(offset)),
        jumpsize(std::forward<JumpsizeType>(jumpsize))
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
        ContainerPtr,  
        ComponentPtr& componentPtr,
        IndexType &,
        RangeType const & distance)
    {
        RangeType counter = 0;
        for(; counter<distance; ++counter)
        {
            for(decltype(jumpsize()) i=0; i< jumpsize(); ++i)
            {
                componentPtr = componentPtr->next;
                if(componentPtr == nullptr)
                    break;
            }
            if(componentPtr == nullptr)
                    break;
        }
        // we need the distance from the last element to the current index position
        return distance - counter;
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
        ContainerPtr,  
        ComponentPtr& componentPtr,
        IndexType &,
        RangeType distance)
    {

        std::cout << "The wrong one" << std::endl;
        RangeType counter = 0;
        for(; counter<distance; ++counter)
        {
            for(decltype(jumpsize()) i=static_cast<decltype(jumpsize())>(0); i< jumpsize(); ++i)
            {
                componentPtr = componentPtr->previous;
                if(componentPtr == nullptr)
                {
                    break;
                }
            }
            if(componentPtr == nullptr)
                    break;
        }
        // we need the distance from the last element to the current index position
        return distance - counter;
    }
    
    /**
     * @brief set the iterator to the first element
     * 
     */

    HDINLINE 
    void 
    begin(
        ContainerPtr containerPtr,  
        ComponentPtr& componentPtr,
        IndexType &)
    {
        componentPtr = containerPtr->first;
    }

    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     */

    HDINLINE 
    void 
    rbegin(
        ContainerPtr containerPtr,  
        ComponentPtr& componentPtr,
        IndexType &)
    {
        auto nbElements = size(containerPtr);
        auto jumps = (nbElements % jumpsize()) - ((nbElements - offset()) % jumpsize()); 
        componentPtr = containerPtr->last;

        for( auto i=0u; i<jumps; ++i)
        {
            if(componentPtr == nullptr)
                break;
            componentPtr = componentPtr->previous;
        }
    }
    
    HDINLINE 
    void 
    end(
        ContainerPtr,  
        ComponentPtr& componentPtr,
        IndexType &)
    {
        componentPtr = nullptr;
    }
    
    /**
     * @brief set the iterator to the last element. It is possible that two iterators,
     * the first start with begin, the second with last, never meet.
     */

    HDINLINE 
    void 
    rend(
        ContainerPtr,  
        ComponentPtr& componentPtr,
        IndexType &)
    {
        componentPtr = nullptr;
    }
    
    HDINLINE 
    bool
    isAfterLast(
        ContainerPtr,  
        ComponentPtr componentPtr,
        IndexType const & )
    const
    {
        return componentPtr == nullptr;
    }
    
    HDINLINE 
    bool
    isBeforeFirst(
        ContainerPtr,  
        ComponentPtr componentPtr,
        IndexType const &)
    const
    {
        return componentPtr == nullptr;
    }
    
//variables
protected:
    OffsetType offset;
    JumpsizeType jumpsize;
    
    template<typename Container>
    uint_fast32_t size(Container* containerPtr)
    {
        uint_fast32_t counter = 0u;
        auto componentPtr = containerPtr->first;
        while(componentPtr != nullptr)
        {
            ++counter;
            componentPtr = componentPtr->next;
        };
        return counter;
    }
};

/**
 * @brief This navigator is a concept. It has an offset and a jumpsize.
 */

template<
    typename TOffset,
    typename TJumpsize>
struct Navigator<
        details::UndefinedType,
        TOffset,
        TJumpsize,
        details::UndefinedType,
        details::UndefinedType>
{
    typedef TOffset OffsetType;
    typedef TJumpsize JumpsizeType;
    typedef details::UndefinedType ContainerType;
    
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
        offset(std::forward<TOffset_>(offset)),
        jumpsize(std::forward<TJumpsize_>(jumpsize))
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
auto 
makeNavigator(
    TOffset && offset,
    TJumpsize && jumpsize)
-> 
    hzdr::Navigator<
        details::UndefinedType,
        typename std::decay<TOffset>::type ,
        typename std::decay<TJumpsize>::type,
        details::UndefinedType,
        details::UndefinedType>
{
    typedef typename std::decay<TOffset>::type OffsetType;
    typedef typename std::decay<TJumpsize>::type JumpsizeType;
    typedef hzdr::Navigator<
        details::UndefinedType,
        OffsetType,
        JumpsizeType,
        details::UndefinedType,
        details::UndefinedType> ResultType;
    
    return ResultType(
        std::forward<TOffset>(offset),
        std::forward<TJumpsize>(jumpsize));
}



namespace details
{



    template<typename T>
    struct NavigatorTemplates
    {
        typedef typename std::decay<T>::type _T;
        typedef typename _T::ContainerType ContainerType;
        typedef typename _T::OffsetType OffsetType;
        typedef typename _T::JumpsizeType JumpsizeType;
    };
    
    

template<
    typename TContainer,
    typename TNavigator,
    typename TOffset = typename details::NavigatorTemplates<TNavigator>::OffsetType,
    typename TJumpsize = typename details::NavigatorTemplates<TNavigator>::JumpsizeType>
auto
makeNavigator(
    TNavigator && navi)
->
hzdr::Navigator<
        typename std::decay<TContainer>::type,
        TOffset,
        TJumpsize,
        typename traits::ContainerCategory<
            typename std::decay<TContainer>::type>::type>
{
    typedef typename std::decay<TContainer>::type                   ContainerType;
    typedef typename traits::ContainerCategory<ContainerType>::type ContainerCategoryType;

    typedef hzdr::Navigator<
        ContainerType,
        TOffset,
        TJumpsize,
        ContainerCategoryType> ResultType;
    return ResultType(std::forward<TOffset>(navi.offset), std::forward<TJumpsize>(navi.jumpsize));
}


} // namespace details



/**
 * @brief bind a container to a navigator concept 
 * @param container 
 * @param navi The concept of a navigator
 */
// template<
//     typename TContainer,
//     typename TOffset,
//     typename TJumpsize>
// auto
// makeNavigator(
//     TContainer &&,
//     hzdr::Navigator<
//         details::UndefinedType,
//         TOffset,
//         TJumpsize,
//         details::UndefinedType,
//         details::UndefinedType> && navi)
// ->
// hzdr::Navigator<
//         typename std::decay<TContainer>::type,
//         TOffset,
//         TJumpsize,
//         typename traits::ContainerCategory<
//             typename std::decay<TContainer>::type>::type>
// {
//     typedef typename std::decay<TContainer>::type                   ContainerType;
//     typedef typename traits::ContainerCategory<ContainerType>::type ContainerCategoryType;
// 
//     typedef hzdr::Navigator<
//         ContainerType,
//         TOffset,
//         TJumpsize,
//         ContainerCategoryType> ResultType;
//     return ResultType(
//         std::forward<TOffset>(navi.offset),
//         std::forward<TJumpsize>(navi.jumpsize));
// }


/**
 * @brief creates an iterator
 * @tparam container type of the container
 * @param offset distance from the start of the container to the first element 
 * of the iterator
 * @param jumpsize distance between to elements within the container
 */
template<
    typename TContainer,
    typename TOffset,
    typename TJumpsize>
auto 
makeNavigator(
    TOffset && offset,
    TJumpsize && jumpsize)
-> 
    hzdr::Navigator<
        typename std::decay<TContainer>::type,
        typename std::decay<TOffset>::type ,
        typename std::decay<TJumpsize>::type,
        typename traits::ContainerCategory<
            typename std::decay<TContainer>::type>::type>
{
    typedef typename std::decay<TContainer>::type               ContainerType;
    typedef typename std::decay<TOffset>::type                  OffsetType;
    typedef typename std::decay<TJumpsize>::type                JumpsizeType;
    
    typedef typename traits::ContainerCategory<ContainerType>::type ContainerCategoryType;
    typedef hzdr::Navigator<
        ContainerType,
        OffsetType,
        JumpsizeType,
        ContainerCategoryType> ResultType;
    
    return ResultType(
        std::forward<TOffset>(offset),
        std::forward<TJumpsize>(jumpsize));
}

}// namespace hzdr

