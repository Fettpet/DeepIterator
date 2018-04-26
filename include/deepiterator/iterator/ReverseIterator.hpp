/* Copyright 2018 Sebastian Hahn

 * This file is part of ReverseDeepIterator.
 *
 * ReverseDeepIterator is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ReverseDeepIterator is distributed in the hope that it will be useful,
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
#include "deepiterator/definitions/forward.hpp"
#include "deepiterator/definitions/NoChild.hpp"
#include "deepiterator/DeepIterator.hpp"
#include <limits>
#include <cassert>
#include <type_traits>
#include <sstream>
#include <typeinfo>

namespace deepiterator 
{
namespace details 
{
/**
 * These four structs are used in the View to call the right constructor. Also 
 * they are needed to construct the iterator.
 */
namespace constructorType
{
struct rbegin{};
struct rend{};
}
}

/**
 * \struct ReverseDeepIterator
 * @author Sebastian Hahn
 * 
 * @brief The ReverseDeepIterator class is used to iterator over interleaved data 
 * structures. The simplest example for an interleaved data structure is 
 * std::vector< std::vector< int > >. The ReverseDeepIterator iterates over all ints 
 * within the structure. 
 * 
 * Inside the ReverseDeepIterator are two variables. These are importent for the 
 * templates. These Variables are:
 * 1. containerPtr: is the pointer to the container, given by the constructor
 * 2. index: the current element index within the container.
 * 
 * 
 * @tparam TContainer : This one describes the container, over whose elements 
 * you would like to iterate. This template need a specialication of the trait
 * ComponentType for TContainer. This trait is the type of the components 
 * of TContainer; 
 * @tparam TAccessor The accessor descripe the access to and position of the 
 * components of TContainer. 
   @tparam TNavigator The navigator describe the way to walk through the data. 
   It describe the first element, the next element and the after last element.
   \see Navigator.hpp
 
   @tparam TChild The child is the template parameter to realize nested 
   structures.  This template has several 
   requirements: 
    1. it need to spezify an Iterator type. These type need operator++,  operator*,
        operator=, operator== and a default constructor.
    2. gotoNext(), nbElements(), setToBegin(), isAfterLast()
    3. gotoPrevious(), setToRbegin(), isBeforeFirst()
    3. TChild::ReturnType must be specified. This is the componenttype of the 
    innerst container.
    4. TChild::IsRandomAccessable The child is random accessable
    5. TChild::IsBidirectional The child is bidirectional
    6. TChild::hasConstantSize The container of the child has constant size
    7. default constructor
    8. copy constructor
    9. constructor with childtype && containertype as variables
   It it is recommended to use ReverseDeepIterator as TChild.
   @tparam TIndexType Type of the index. The index is used to access the component 
   within the container. The index must support a cast from int especially from
   0.
   @tparam hasConstantSizeSelf This flag is used to decide whether the container
   has a fixed number of elements. It is not needed that this count is known at 
   compiletime, but recommended. This trait is used to optimize the iteration to
   the next element. At default, a container hasnt a fixed size. An example 
   for a container with fixed size is std::array<T, 10>. 
   @tparam isBidirectionalSelf This flag is used to decide wheter the container
   is bidirectional. If this flag is set to true, it enables backward iteration
   i. e. operator--. The navigator need the bidirectional functions
   @tparam isRandomAccessableSelf This flag is used to decide whether the 
   container is random accessable. If this flag is set to true, it enables the 
   following operations: +, +=, -, -=, <,>,>=,<=. The accessor need the functions
   lesser, greater, if this flag is set to true.
   \see Componenttype.hpp Accessor.hpp
 # Implementation details{#sectionD2}
The ReverseDeepIterator supports up to four constructors: begin, end, rbegin, rend. To 
get the right one, we had four classes in details::constructorType. The 
constructors has five parameters: 
    ContainerPtr container, 
    TAccessor && accessor, 
    TNavigator && navigator,
    TChild && child,
    details::constructorType::__
If your container has no interleaved layer, use \b deepiterator::NoChild as child.
A ReverseDeepIterator is bidirectional if the flag isBidirectionalSelf is set to true 
and all childs are bidirectional. The same applies to random accessablity.

We had two algorithms inside the ReverseDeepIterator. The first one is used to find the
first element within a nested data structure. The second one is used to find the
next element within the data structure. Lets start with the find the first 
element procedure. The setToBegin function has an optional parameter, this is 
the container over which the iterator walks. The first decision "Has Childs" is
done by the compiler. We had two different ReverseDeepIterators. One for the has childs
case, and one for the has no childs case. The case where the iterator has childs
search now an element, where all childs are valid. It pass the current element
to the child. The child go also the the beginning. If the current element hasnt
enough elements, we iterate one element forward and check again.
\image html images/setTobegin.png "Function to find the first element"
The second algorithm is used to find the previous element. We show this at the
operator--. The operator- calls also the gotoPrevious function, with an other 
value rather than 1. First we check whether they are childs. If not, we call the
navigator. If there are childs, we call gotoPrevious. The gotoPrevious function 
first check whether the iterator has childs, i.e. has an interleaved datastructure.
If it has childs there are two different approches. The first one assumes that 
each element of the container has the same size. The spit the jumpsize in 3 values
1. the rest in this element,
2. the number of elements, which are overjumped,
3. the remainder of the resulting element
In the second case, where each element can have a different number of elements, 
the ReverseDeepIterator doesnt overjump elements. It walks step by step.
\image html images/setTobegin.png
 */
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TChild,
    typename TIndexType,
    bool hasConstantSizeSelf = false,
    bool isBidirectionalSelf = true,
    bool isRandomAccessableSelf = true>
struct ReverseDeepIterator
{
// datatypes
    
public:
    using ContainerType = TContainer;
    using ContainerRef = ContainerType&;
    using ContainerPtr = ContainerType*;
    
    using Accessor = TAccessor;
    using Navigator = TNavigator;
    
// child things
    using ComponentType = typename deepiterator::traits::ComponentType<
        ContainerType
    >::type;
    
    using ChildIterator = TChild;
    using ReturnType = typename ChildIterator::ReturnType;

// container stuff
    using IndexType = TIndexType;
    
protected:
    ContainerPtr containerPtr = nullptr;
    IndexType index;
    ChildIterator childIterator;
    Navigator navigator;
    Accessor accessor;
    
public:
    using RangeType = decltype(((Navigator*)nullptr)->next(
        nullptr,
        index,
        0
    ));
    // decide wheter the iterator is bidirectional.
    static const bool isBidirectional = ChildIterator::isBidirectional && isBidirectionalSelf;
    static const bool isRandomAccessable = ChildIterator::isRandomAccessable && isRandomAccessableSelf;
    
    static const bool hasConstantSizeChild = ChildIterator::hasConstantSize;

    
    static const bool hasConstantSize = hasConstantSizeSelf && hasConstantSizeChild;

    
public:

// The default constructors
    HDINLINE ReverseDeepIterator() = default;
    HDINLINE ReverseDeepIterator(ReverseDeepIterator const &) = default;
    HDINLINE ReverseDeepIterator(ReverseDeepIterator &&) = default;
    
// the default copy operators
    HDINLINE ReverseDeepIterator& operator=(ReverseDeepIterator const &) = default;
    HDINLINE ReverseDeepIterator& operator=(ReverseDeepIterator &&) = default;

    
    /**
     * @brief This constructor is used to create a iterator in a middle layer. 
     * The container must be set with setToBegin or setToRbegin. The iterator 
     * constructed with this function is not valid.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child iterator for the next layer
     */
    template<
        typename TAccessor_, 
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    ReverseDeepIterator(
            TAccessor_ && accessor, 
            TNavigator_ && navigator,
            TChild_ && child
    ):
        containerPtr(nullptr),
        index(static_cast<IndexType>(0)),
        childIterator(deepiterator::forward<TChild_>(child)),
        navigator(deepiterator::forward<TNavigator_>(navigator)),
        accessor(deepiterator::forward<TAccessor_>(accessor))
    {}
    
    
    /**
     * @brief This constructor is used to create an iterator at the rbegin element.
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child iterator for the next layer
     * @param details::constructorType::begin used to specify that the begin 
     * element is needed. We recommend details::constructorType::rbegin() as 
     * parameter.
     */
    template<
        typename TAccessor_,
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    ReverseDeepIterator(
            ContainerPtr container, 
            TAccessor_&& accessor, 
            TNavigator_&& navigator,
            TChild_&& child,
            details::constructorType::rbegin
    ):
        containerPtr(container),
        index(static_cast<IndexType>(0)),
        childIterator(deepiterator::forward<TChild_>(child)),
        navigator(deepiterator::forward<TNavigator_>(navigator)),
        accessor(deepiterator::forward<TAccessor_>(accessor))
    {
        setToRbegin(container);
    }
    
    
    template<
        typename TPrescription
    >
    HDINLINE
    ReverseDeepIterator(
            ContainerPtr container, 
            TPrescription&& prescription,
            details::constructorType::rbegin
    ):
        containerPtr(container),
        index(static_cast<IndexType>(0)),
        childIterator(
            deepiterator::forward<TPrescription>(prescription).child, 
            details::constructorType::rbegin()
        ),
        navigator(deepiterator::forward<TPrescription>(prescription).navigator),
        accessor(deepiterator::forward<TPrescription>(prescription).accessor)
    {
        setToRbegin(container);
    }
    
    
    template<typename TPrescription_>
    HDINLINE
    ReverseDeepIterator( 
            TPrescription_ && prescription, 
            details::constructorType::rbegin
    ):
        containerPtr(nullptr),
        index(0),
        childIterator(
            deepiterator::forward<TPrescription_>(prescription).child,
            details::constructorType::rbegin()
        ),
        navigator(deepiterator::forward<TPrescription_>(prescription).navigator),
        accessor(deepiterator::forward<TPrescription_>(prescription).accessor)
    {}
    
    
    /**
     * @brief This constructor is used to create an iterator at the rend element.
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child iterator for the next layer
     * @param details::constructorType::rend used to specify that the end
     * element is needed. We recommend details::constructorType::rend() as 
     * parameter.
     */
    template<
        typename TAccessor_,
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    ReverseDeepIterator(
            ContainerPtr container, 
            TAccessor_&& accessor, 
            TNavigator_&& navigator,
            TChild_&& child,
            details::constructorType::rend
    ):
        containerPtr(container),
        index(static_cast<IndexType>(0)),
        childIterator(deepiterator::forward<TChild_>(child)),
        navigator(deepiterator::forward<TNavigator_>(navigator)),
        accessor(deepiterator::forward<TAccessor_>(accessor))
        
    {
        setToRend(container);
    }

    
    template<
        typename TPrescription
    >
    HDINLINE
    ReverseDeepIterator(
            ContainerPtr container, 
            TPrescription&& prescription,
            details::constructorType::rend 
    ):
        containerPtr(container),
        index(static_cast<IndexType>(0)),
        childIterator(
            deepiterator::forward<TPrescription>(prescription).child, 
            details::constructorType::rend()
        ),
        navigator(deepiterator::forward<TPrescription>(prescription).navigator),
        accessor(deepiterator::forward<TPrescription>(prescription).accessor)
    {
        setToRend(container);
    }
    
    
    template<typename TPrescription_>
    HDINLINE
    ReverseDeepIterator(
            TPrescription_ && prescription, 
            details::constructorType::rend
    ):
        containerPtr(nullptr),
        index(0),
        childIterator(
            deepiterator::forward<TPrescription_>(prescription).child,
            details::constructorType::rend()
        ),
        navigator(deepiterator::forward<TPrescription_>(prescription).navigator),
        accessor(deepiterator::forward<TPrescription_>(prescription).accessor)
    {}
    
    
    /**
     * @brief grants access to the current elment. This function calls the * 
     * operator of the child iterator. The behavior is undefined, if the iterator 
     * would access an element out of the container.
     * @return the current element.
     */
    HDINLINE
    auto
    operator*()
    ->
    ReturnType
    {
        return *childIterator;
    }
    
    
    /**
     * @brief compares the ReverseDeepIterator with an other ReverseDeepIterator.
     * @return true: if the iterators are at different positions, false
     * if they are at the same position
     */
    HDINLINE
    bool
    operator!=(const ReverseDeepIterator& other)
    const
    {
        return not (*this == other);
    }
    
    /**
     * @brief grants access to the current elment. This function calls the * 
     * operator of the child iterator. The behavior is undefined, if the iterator 
     * would access an element out of the container.
     * @return the current element.
     */
    HDINLINE 
    ReturnType
    operator->()
    {
        return *childIterator;
    }

    /**
     * @brief compares the ReverseDeepIterator with an other ReverseDeepIterator.
     * @return false: if the iterators are at different positions, true
     * if they are at the same position
     */
    HDINLINE
    auto
    operator==(const ReverseDeepIterator& other)
    const
    ->
    bool
    {

        return (isAfterLast() && other.isAfterLast())
            || (isBeforeFirst() && other.isBeforeFirst())
            ||(containerPtr == other.containerPtr
            && index == other.index 
            && other.childIterator == childIterator);
    }
    
    /**
     * @brief goto the next element. If the iterator is at the before-first-element
     * it is set to the begin element.
     * @return reference to the next element
     */
    HDINLINE
    auto
    operator--()
    ->
    ReverseDeepIterator&
    {   
        if(isBeforeFirst())
        {
            setToBegin();
            return *this;
        }
        gotoNext(1u);
        return *this;
    }
    
    /**
     * @brief goto the next element. If the iterator is at the before-first-element
     * it is set to the begin element.
     * @return reference to the current element
     */
    HDINLINE
    auto
    operator--(int)
    -> 
    ReverseDeepIterator
    {
        ReverseDeepIterator tmp(*this);
        --tmp;
        return tmp;
    }
    

    /**
     * @brief goto the previous element. If the iterator is at after-first-element,
     * it is set to the rbegin element. The iterator need to be bidirectional to
     * support this function.
     * @return reference to the previous element
     */
    HDINLINE
    auto
    operator++()
    ->
    ReverseDeepIterator&
    {
        // if the iterator is after the last element, we set it to the last 
        // element
        if(isAfterLast())
        {
            setToRbegin();
            return *this;
        }
        gotoPrevious(1u);
        return *this;
    }
    
    /**
     * @brief goto the previous element. If the iterator is at after-first-element,
     * it is set to the rbegin element. The iterator need to be bidirectional to
     * support this function.
     * @return reference to the current element
     */
    HDINLINE
    auto
    operator++(int)
    ->
    ReverseDeepIterator
    {
        ReverseDeepIterator tmp(*this);
        ++(*this);
        return tmp;
    }
    
        /**
     * @brief set the iterator jumpsize elements ahead. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     */
    HDINLINE 
    auto
    operator-=(uint const & jumpsize) 
    -> 
    ReverseDeepIterator&
    {
        auto tmpJump = jumpsize;
        // if the iterator is before the first element, be set it to the first
        if(isBeforeFirst())
        {
            --tmpJump;
            setToBegin();
        }
        gotoNext(tmpJump);
        return *this;

    }
    
    /**
     * @brief the gotoPrevious function has two implementations. The first one 
     * is used if the container of the child has a constant size. This is 
     * implemented here. The second one is used if the container of the child 
     * hasnt a constant size. The cost of this function is O(1).
     * @param jumpsize Distance to the previous element
     * @return The result value is importent if the iterator is in a middle layer.
     * When we reach the end of the container, we need to give the higher layer
     * informations about the remainig elements, we need to overjump. This distance
     * is the return value of this function.
     */
    template< 
        bool T = hasConstantSizeChild> 
    HDINLINE 
    auto
    gotoPrevious(uint const & jumpsize)
    ->
    typename std::enable_if<
        T == true, 
        uint
    >::type    
    {
        using SizeChild_t = decltype(childIterator.nbElements());
        using ResultType_t = decltype(navigator.previous(
            containerPtr,
            index, 
            0u
        ));
        /** 
         * For implementation details see gotoNext
         */
        auto && childNbElements = childIterator.nbElements();        
        if(childNbElements == static_cast<SizeChild_t>(0))
        {
            setToRend(containerPtr);
            return 0u;
        }
        
        int && remaining = childIterator.gotoPrevious(jumpsize);
        auto && overjump{(remaining + childNbElements - 1) / childNbElements};
        auto && childJumps{((remaining - 1) % childNbElements)};

        
        ResultType_t const result{navigator.previous(
            containerPtr, 
            index, 
            overjump
        )};
        if((result == static_cast<ResultType_t>(0)) && (overjump > 0))
        {

                childIterator.setToRbegin(accessor.get(
                    containerPtr, 
                    index
                ));
                childIterator.gotoPrevious(
                    childJumps
                );

        }
        // we only need to return something, if we are at the end
        auto const condition = (result > static_cast<ResultType_t>(0u));
        // the size of the jumps
        uint const notOverjumpedElements = 
                (result-static_cast<ResultType_t>(1u)) * childNbElements;
        
        // The 1 is to set to the first element
        return condition * (
            notOverjumpedElements 
          + childJumps 
          + static_cast<ResultType_t>(1u)
        );
    }
    
    /**
     * @brief the gotoPrevious function has two implementations. The first one 
     * is used if the container of the child has a constant size. The second one 
     * is used if the container of the child hasnt a constant size. This is 
     * implemented here. The function, we go in the child to the end, go to the 
     * previos element and repeat this procedure until we had enough jumps. This 
     * is an expensive procedure.
     * @param jumpsize Distance to the next element
     * @return The result value is importent if the iterator is in a middle layer.
     * When we reach the end of the container, we need to give the higher layer
     * informations about the remainig elements, we need to overjump. This distance
     * is the return value of this function.
     */
    template< bool T = hasConstantSizeChild> 
    HDINLINE 
    auto
    gotoPrevious(uint const & jumpsize)
    -> 
    typename std::enable_if<
        T == false, 
        uint
    >::type 
    {
        auto remaining = jumpsize;
        while(remaining > 0u && not isBeforeFirst())
        {
            if(not childIterator.isBeforeFirst())
            {
                remaining = childIterator.gotoPrevious(remaining);
                if(remaining == 0u)
                    break;
                --remaining;
            }
            while(childIterator.isBeforeFirst() && not isBeforeFirst())
            {
                navigator.previous(
                    containerPtr, 
                    index, 
                    1u
                );
                if(not isBeforeFirst())
                    childIterator.setToRbegin(accessor.get(
                        containerPtr, 
                        index
                    ));
            }
        }
        return remaining;
    }
    
    
    /**
     * @brief creates an iterator which is jumpsize elements behind. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the previous element
     * @return iterator which is jumpsize elements behind
     */
    HDINLINE 
    auto
    operator-(uint const & jumpsize)
    -> 
    ReverseDeepIterator
    {
        ReverseDeepIterator tmp(*this);
        tmp -= jumpsize;
        return tmp;
    }


    /**
     * @brief set the iterator jumpsize elements behind. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     */
    HDINLINE 
    auto 
    operator+=(const uint & jumpsize)
    ->
    ReverseDeepIterator&
    {
        auto tmpJump = jumpsize;
        if(isAfterLast())
        {
            --tmpJump;
            setToRbegin();
        }
        gotoPrevious(tmpJump);
        return *this;
    }

    
    /**
     * @brief the gotoNext function has two implementations. The first one is used
     * if the container of the child has a constant size. This is implemented 
     * here. The second one is used if the container of the child hasnt a 
     * constant size. The cost of this function is O(1).
     * @param jumpsize Distance to the next element
     * @return The result value is importent if the iterator is in a middle layer.
     * When we reach the end of the container, we need to give the higher layer
     * informations about the remainig elements, we need to overjump. This distance
     * is the return value of this function.
     */
    template< bool T = hasConstantSizeChild> 
    HDINLINE 
    auto
    gotoNext(uint const & jumpsize)
    ->    
    typename std::enable_if<
        T == true, 
        uint
    >::type 
    {
        /**
         * The variable jumpsize is compond from three other variables:
         * 1. The distance of the child to these end
         * 2. the number of childs we can overjump
         * 3. the remaining jumpsize for the new child
         */
        
        // get the number of elements and overjump, if it has not enough 
        // elements
        auto && childNbElements = childIterator.nbElements();        
        
        if(childNbElements == 0)
        {
            setToEnd(containerPtr);
            return 0;
        }
        
        auto && remaining = childIterator.gotoNext(jumpsize);
        
        // the -1 is used, since we jump from an end to the begining of the next cell
        auto && overjump = (remaining - 1 + childNbElements) / childNbElements;
        int childJumps = ((remaining - 1) % childNbElements);
        
        int && result = navigator.next(containerPtr, index, overjump);
        // result == 0 means the point lays within this data structure
        // overjump > 0 means we change the datastructure
        if((result == 0) && (overjump > 0))
        {
            childIterator.setToBegin(accessor.get(
                containerPtr, 
                index
            ));
            childIterator.gotoNext(childJumps);
        }
        // we only need to return something, if we are at the end
        uint const condition = (result > 0);
        // the size of the jumps
        uint const notOverjumpedElements = (result-1) * childNbElements;
        
        // The 1 is to set to the first element
        return condition * (notOverjumpedElements + childJumps + 1u);
    }
    
    /**
     * @brief the gotoNext function has two implementations. The first one is used
     * if the container of the child has a constant size. The second one is used
     * if the container of the child hasnt a constant size. This is implemented 
     * here. The function, we go in the child to the end, go to the next element
     * and repeat this procedure until we had enough jumps. This is an expensive
     * procedure.
     * @param jumpsize Distance to the next element
     * @return The result value is importent if the iterator is in a middle layer.
     * When we reach the end of the container, we need to give the higher layer
     * informations about the remainig elements, we need to overjump. This distance
     * is the return value of this function.
     */
    template<bool T = hasConstantSizeChild> 
    HDINLINE 
    auto 
    gotoNext(uint const & jumpsize)
    -> 
    typename std::enable_if<
        T == false, 
        uint
    >::type 
    {
        // we need to go over all elements
        auto remaining = jumpsize;
        while(remaining > 0u && not isAfterLast())
        {
            if(not childIterator.isAfterLast())
            {
                // we go to the right element, or the end of this container
                remaining = childIterator.gotoNext(remaining);
                // we have found the right element
                if(remaining == 0u)
                    break;
                // we go to the next container
                --remaining;
            }
            while(childIterator.isAfterLast() && not isAfterLast())
            {
                navigator.next(containerPtr, index, 1u);
                // only valid, if it contains enough elements
                if(not isAfterLast())
                    childIterator.setToBegin(accessor.get(containerPtr, index));
            }
        }
        return remaining;
    }
    
    /**
     * @brief check whether the iterator is behind a second one.
     * @return true if the iterator is behind, false otherwise
     */
    template< bool T=isRandomAccessable>
    HDINLINE
    auto
    operator>(ReverseDeepIterator const & other)
    -> 
    typename std::enable_if<
        T == true, 
        bool
    >::type
    {
        if(accessor.lesser(
                containerPtr, 
                index,
                other.containerPtr,
                other.index
            )
        )
           return true;
        if( accessor.equal(
                containerPtr, 
                index,
                other.containerPtr,
                other.index
            ) && childIterator < other.childIterator
        )
            return true;
        return false;
    }
    
    /**
     * @brief check whether the iterator is ahead a second one.
     * @return true if the iterator is ahead, false otherwise
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto
    operator<(ReverseDeepIterator const & other)
    -> 
    typename std::enable_if<
        T == true, 
        bool
    >::type
    {
        if(accessor.greater(
                containerPtr, 
                index,
                other.containerPtr,
                other.index
            )
        )
           return true;
        
        if(accessor.equal(
                containerPtr, 
                index,
                other.containerPtr,
                other.index
            ) 
            &&
            childIterator < other.childIterator
        )
            return true;
            
        return false;
    }
    
            /**
     * @brief check whether the iterator is behind or equal a second one.
     * @return true if the iterator is behind or equal, false otherwise
     */
    template< bool T=isRandomAccessable>
    HDINLINE
    auto 
    operator<=(ReverseDeepIterator const & other) 
    ->    
    typename std::enable_if<
        T == true, 
        bool
    >::type
    {
        return *this < other || *this == other;
    }
    
    
    /**
     * @brief check whether the iterator is ahead or equal a second one.
     * @return true if the iterator is ahead or equal, false otherwise
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto 
    operator>=(ReverseDeepIterator const & other) 
    -> 
    typename std::enable_if<
        T == true, 
        bool
    >::type
    {
        return *this > other || *this == other;
    }
    
    
    /**
     * @return get the element at the specified position.
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto 
    operator[](IndexType const & index)
    -> 
    typename std::enable_if<
        T == true, 
        ReturnType&
    >::type
    {
        ReverseDeepIterator tmp(*this);
        tmp.setToBegin();
        tmp += index;
        return *tmp;
    }
    
    
    /**
     * @brief This function set the iterator to the first element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element. 
     */
    HDINLINE
    auto 
    setToBegin()
    -> 
    void
    {
        navigator.begin(
            containerPtr, 
            index
        );
        // check whether the iterator is at a valid element
        while(not isAfterLast())
        {
            childIterator.setToBegin((accessor.get(
                containerPtr, 
                index
            )));
            if(not childIterator.isAfterLast())
                break;
            gotoNext(1u);
        }
    }
    

    /**
     * @brief This function set the iterator to the first element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     */
    HDINLINE
    auto
    setToBegin(TContainer& con)
    -> 
    void
    {
        containerPtr = &con;
        setToBegin();

    }
    
    /**
     * @brief This function set the iterator to the first element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     */
    HDINLINE
    auto 
    setToBegin(TContainer* ptr)
    -> 
    void
    {
        containerPtr = ptr;
        setToBegin();
    }
    
    /**
     * @brief This function set the iterator to the after-last-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     * */
    HDINLINE
    auto
    setToEnd(TContainer* ptr)
    -> 
    void
    {
        containerPtr = ptr;
        navigator.end(
            containerPtr, 
            index
        );
    }
    
    /**
     * @brief This function set the iterator to the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     * */
    HDINLINE
    auto
    setToRend(TContainer* ptr)
    -> 
    void
    {
        containerPtr = ptr;
        navigator.rend(
            containerPtr,
            index
        );
    }

        /**
     * @brief This function set the iterator to the last element. This function
     * set also all childs to rbegin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     */
    HDINLINE
    auto 
    setToRbegin()
    -> 
    void
    {
        navigator.rbegin(
            containerPtr, 
            index
        );
        
        // check whether the iterator is at a valid element
        while(not isBeforeFirst())
        {
            childIterator.setToRbegin((accessor.get(
                containerPtr, 
                index
            )));
            if(not childIterator.isBeforeFirst())
                break;
            gotoPrevious(1u);
        }
    }
    
    /** 
     * @brief This function set the iterator to the last element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     */
    HDINLINE
    auto 
    setToRbegin(ContainerRef con)
    -> 
    void
    {
        containerPtr = &con;
        setToRbegin();
    }
    
    /**
     * @brief This function set the iterator to the last element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     */
    HDINLINE
    auto  
    setToRbegin(ContainerPtr ptr)
    -> 
    void
    {
        containerPtr = ptr;
        setToRbegin();
    }

    /**
     * @brief check whether the iterator is after the last element
     * @return true, if it is, false if it is not after the last element
     */
    HDINLINE 
    auto
    isAfterLast()
    const
    -> 
    bool
    {
        return navigator.isAfterLast(
            containerPtr, 
            index
        );
    }
    
    /**
     * @brief check whether the iterator is before the first element
     * @return true, if it is, false if it is not after the last element
     */
    HDINLINE 
    auto
    isBeforeFirst()
    const
    -> 
    bool
    {
        return navigator.isBeforeFirst(
            containerPtr, 
            index
        );
    }
    
    /**
     * @brief if the container has a constant size, this function can caluculate
     * it.
     * @return number of elements within the container. This include the child
     * elements
     */
    template<bool T = hasConstantSize>
    HDINLINE
    auto
    nbElements()
    const
    -> 
    typename std::enable_if<
        T == true, 
        int
    >::type
    {
        return childIterator.nbElements() * navigator.size(containerPtr);
    }
    
    HDINLINE
    ContainerPtr
    getContainerPtr()
    const
    {
        return containerPtr;
    }
    
    HDINLINE
    IndexType const &
    getIndex()
    const
    {
        return index;
    }

    HDINLINE
    ChildIterator const &
    getChild()
    const 
    {
        return childIterator;
    }
} ; // struct ReverseDeepIterator





/** ************************************+
 * @brief The flat implementation. This ReverseDeepIterator has no childs. 
 * ************************************/
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator,
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf>
struct ReverseDeepIterator<
    TContainer,     
    TAccessor, 
    TNavigator,
    deepiterator::NoChild,
    TIndexType,
    hasConstantSizeSelf,
    isBidirectionalSelf,
    isRandomAccessableSelf
>
{
// datatypes
public:

    using ContainerType = TContainer;
    using ContainerRef = ContainerType&;
    using ContainerPtr = ContainerType*;
    
    using Accessor = TAccessor;
    using Navigator = TNavigator;
    
// child things
    using ComponentType = typename deepiterator::traits::ComponentType<
        ContainerType
    >::type;
    using ComponentPtr = ComponentType*;
    using ComponentReference = ComponentType&;
    using ChildIterator = deepiterator::NoChild;
    using ReturnType = ComponentReference;

// container stuff
    using IndexType = TIndexType;


// Variables    
protected:   
    Navigator navigator;
    Accessor accessor;
    deepiterator::NoChild childIterator;
    

    ContainerType* containerPtr;
    IndexType index;
// another data types
public:
    using RangeType = decltype(((Navigator*)nullptr)->next(
        nullptr,
        index,
        0
    ));
// container stuff

    
    static const bool isBidirectional = isBidirectionalSelf;
    static const bool isRandomAccessable = isRandomAccessableSelf;
    static const bool hasConstantSize = hasConstantSizeSelf;
// functions 
public:


    HDINLINE ReverseDeepIterator() = default;
    HDINLINE ReverseDeepIterator(const ReverseDeepIterator&) = default;
    HDINLINE ReverseDeepIterator(ReverseDeepIterator&&) = default;

    HDINLINE ReverseDeepIterator& operator=(ReverseDeepIterator const &) = default;
    HDINLINE ReverseDeepIterator& operator=(ReverseDeepIterator &&) = default;
    
    
     /**
     * @brief This constructor is used to create an iterator at the begin element.
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child Use NoChild()
     * @param details::constructorType::begin used to specify that the begin
     * element is needed. We recommend details::constructorType::begin() as 
     * parameter.
     */
    template<
        typename TAccessor_,
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    ReverseDeepIterator(
            ContainerPtr container, 
            TAccessor_&& accessor, 
            TNavigator_&& navigator,
            TChild_ const &,
            details::constructorType::rbegin
    ):
        navigator(deepiterator::forward<TNavigator_>(navigator)),
        accessor(deepiterator::forward<TAccessor_>(accessor)),
        childIterator(),
        containerPtr(container),
        index(static_cast<IndexType>(0))
    {
        setToRbegin(container);
    }
    
    
    template<
        typename TPrescription
    >
    HDINLINE
    ReverseDeepIterator(
            ContainerPtr container, 
            TPrescription&& prescription,
            details::constructorType::rbegin 
    ):
        navigator(deepiterator::forward<TPrescription>(prescription).navigator),
        accessor(deepiterator::forward<TPrescription>(prescription).accessor),
        childIterator(),
        containerPtr(container),
        index(static_cast<IndexType>(0))
    {
        setToRbegin(container);
    }
    
    
    /**
     * @brief This constructor is used to create an iterator at the begin 
     * element. We use this constructor for hierachicale data structures
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child Use NoChild()
     * @param details::constructorType::begin used to specify that the begin
     * element is needed. We recommend details::constructorType::begin() as 
     * parameter.
     */
    template< typename TPrescription_>
    HDINLINE
    ReverseDeepIterator(
            TPrescription_&& prescription, 
            details::constructorType::rbegin 
    ):
        navigator(deepiterator::forward<TPrescription_>(prescription).navigator),
        accessor(deepiterator::forward<TPrescription_>(prescription).accessor),
        childIterator(),
        containerPtr(nullptr),
        index(static_cast<IndexType>(0))
    {}
    
    
    /**
     * @brief This constructor is used to create an iterator at the end element.
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child Use NoChild()
     * @param details::constructorType::end used to specify that the end
     * element is needed. We recommend details::constructorType::end() as 
     * parameter.
     */
    template<
        typename TAccessor_,
        typename TNavigator_,
        typename TChild_
    >
    HDINLINE
    ReverseDeepIterator( 
        TAccessor_ && accessor,
        TNavigator_ && navi,
        TChild_ && 
    ):
        navigator(deepiterator::forward<TNavigator_>(navi)),
        accessor(deepiterator::forward<TAccessor_>(accessor)),
        childIterator()
    {}
    
    
    /**
     * @brief This constructor is used to create an iterator at the rend element.
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child Use NoChild()
     * @param details::constructorType::rend used to specify that the rend
     * element is needed. We recommend details::constructorType::end() as 
     * parameter.
     */
    template<
        typename TAccessor_,
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    ReverseDeepIterator(
            ContainerPtr container, 
            TAccessor_&& accessor, 
            TNavigator_&& navigator,
            TChild_ const &,
            details::constructorType::rend
    ):
        navigator(deepiterator::forward<TNavigator_>(navigator)),
        accessor(deepiterator::forward<TAccessor_>(accessor)),
        containerPtr(container),
        index(static_cast<IndexType>(0))
    {
        setToRend(container);
    }
    
    
    template<
        typename TPrescription
    >
    HDINLINE
    ReverseDeepIterator(
            ContainerPtr container, 
            TPrescription&& prescription,
            details::constructorType::rend 
    ):
        navigator(deepiterator::forward<TPrescription>(prescription).navigator),
        accessor(deepiterator::forward<TPrescription>(prescription).accessor),
        childIterator(),
        containerPtr(container),
        index(static_cast<IndexType>(0))
    {
        setToRend(container);
    }
    
    
    /**
     * @brief This constructor is used to create an iterator at the begin 
     * element. We use this constructor for hierachicale data structures
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child Use NoChild()
     * @param details::constructorType::begin used to specify that the begin
     * element is needed. We recommend details::constructorType::begin() as 
     * parameter.
     */
    template< typename TPrescription_>
    HDINLINE
    ReverseDeepIterator(
            TPrescription_&& prescription, 
            details::constructorType::rend
    ):
        navigator(deepiterator::forward<TPrescription_>(prescription).navigator),
        accessor(deepiterator::forward<TPrescription_>(prescription).accessor),
        childIterator(),
        containerPtr(nullptr),
        index(0)
    {}
    
    
    /**
     * @brief This constructor is used to create a iterator in a middle layer. 
     * The container must be set with setToBegin or setToRbegin.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child use deepiterator::NoChild()
     */
    template<
        typename TAccessor_, 
        typename TNavigator_>
    HDINLINE
    ReverseDeepIterator(
            TAccessor_ && accessor, 
            TNavigator_ && navigator,
            deepiterator::NoChild const &
    ):
        navigator(deepiterator::forward<TNavigator_>(navigator)),
        accessor(deepiterator::forward<TAccessor_>(accessor)),
        containerPtr(nullptr),
        index(0)
    {}
    
    
    /**
     * @brief goto the next element. If the iterator is at the before-first-
     * element it is set to the begin element.
     * @return reference to the next element
     */
    HDINLINE
    auto
    operator--()
    -> 
    ReverseDeepIterator&
    {
        if(isBeforeFirst())
        {
            setToBegin();
        }
        else 
        {
            navigator.next(
                containerPtr, 
                index,
                1u
            );
        }
        return *this;
    }
    
    
    /**
     * @brief goto the next element. If the iterator is at the before-first-
     * element it is set to the begin element.
     * @return reference to the current element
     */
    HDINLINE
    auto 
    operator++(int)
    -> 
    ReverseDeepIterator
    {
        ReverseDeepIterator tmp(*this);
        ++(*this);
        return tmp;
    }
    
    
    /**
     * @brief grants access to the current elment. The behavior is undefined, if
     * the iterator would access an element out of the container.
     * @return the current element.
     */
    HDINLINE
    auto 
    operator*()
    -> 
    ComponentReference
    {
        return accessor.get(
            containerPtr, 
            index
        );
    }
    
    
    /**
     * @brief grants access to the current elment. The behavior is undefined, if
     * the iterator would access an element out of the container.
     * @return the current element.
     */
    HDINLINE
    auto 
    operator->()
    -> 
    ComponentReference
    {
        return accessor.get(
            containerPtr, 
            index
        );
    }
    
    
    /**
     * @brief compares the ReverseDeepIterator with an other ReverseDeepIterator.
     * @return true: if the iterators are at different positions, false
     * if they are at the same position
     */
    HDINLINE
    auto
    operator!=(const ReverseDeepIterator& other)
    const
    -> 
    bool
    {
        
        return (containerPtr != other.containerPtr
             || index != other.index)
             && (not isAfterLast() || not other.isAfterLast())
             && (not isBeforeFirst() || not other.isBeforeFirst());
    }
    
    
    /**
     * @brief compares the ReverseDeepIterator with an other ReverseDeepIterator.
     * @return false: if the iterators are at different positions, true
     * if they are at the same position
     */
    HDINLINE
    auto
    operator==(const ReverseDeepIterator& other)
    const
    -> 
    bool
    {
        return not (*this != other);
    }
    
    
    /**
     * @brief goto the previous element. If the iterator is at after-first-
     * element, it is set to the rbegin element. The iterator need to be 
     * bidirectional to support this function.
     * @return reference to the previous element
     */
    HDINLINE 
    auto 
    operator++()
    -> 
    ReverseDeepIterator& 
    {
        navigator.previous(
            containerPtr, 
            index, 
            1u
        );
        return *this;
    }
    
    
    /**
     * @brief goto the previous element. If the iterator is at after-first-
     * element, it is set to the rbegin element. The iterator need to be 
     * bidirectional to support this function.
     * @return reference to the current element
     */
    HDINLINE 
    auto 
    operator--(int)
    -> 
    ReverseDeepIterator
    {
        ReverseDeepIterator tmp(*this);
        --(*this);
        return tmp;
    }
    
    
     /**
     * @brief set the iterator jumpsize elements ahead. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     */
    template<bool T=isRandomAccessable>
    HDINLINE
    auto
    operator-=(RangeType const & jump)
    -> 
    typename std::enable_if<
        T == true, 
        ReverseDeepIterator&
    >::type
    {
        auto tmpJump = jump;
        if(jump != static_cast<RangeType>(0) && isBeforeFirst())
        {
            --tmpJump;
            setToBegin();
        }
        navigator.next(
            containerPtr, 
            index, 
            tmpJump
        );
        return *this;
    }
    
    
    /**
     * @brief set the iterator jumpsize elements behind. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto
    operator+=(RangeType const & jump) 
    -> 
    typename std::enable_if<
        T == true, 
        ReverseDeepIterator&
    >::type
    {
        auto tmpJump = jump;
        if(jump != static_cast<RangeType>(0) && isAfterLast())
        {
            --tmpJump;
            setToRbegin();
        }
        navigator.previous(
            containerPtr,
            index, 
            tmpJump
        );
        return *this;
    }
    
    
    /**
     * @brief creates an iterator which is jumpsize elements ahead. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     * @return iterator which is jumpsize elements ahead
     */
    template<bool T=isRandomAccessable>
    HDINLINE
    auto
    operator+(RangeType const & jump) 
    -> 
    typename std::enable_if<
        T == true, 
        ReverseDeepIterator
    >::type
    {
        ReverseDeepIterator tmp = *this;
        tmp+=jump;
        return tmp;
    }
    
    
    /**
     * @brief creates an iterator which is jumpsize elements behind. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the previos element
     * @return iterator which is jumpsize elements behind
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto
    operator-(RangeType const & jump)
    ->
    typename std::enable_if<
        T == true, 
        ReverseDeepIterator
    >::type
    {
        ReverseDeepIterator tmp = *this;
        tmp-=jump;
        return tmp;
    }
    
    
    /**
     * @brief check whether the iterator is behind a second one.
     * @return true if the iterator is behind, false otherwise
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto
    operator<(ReverseDeepIterator const & other) 
    -> 
    typename std::enable_if<
        T == true, 
        bool
    >::type
    {
        return accessor.lesser(
            containerPtr, 
            index,
            other.containerPtr,
            other.index
        );
    }
    
    
    /**
     * @brief check whether the iterator is ahead a second one.
     * @return true if the iterator is ahead, false otherwise
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto
    operator>(ReverseDeepIterator const & other)
    -> 
    typename std::enable_if<
        T == true, 
        bool
    >::type
    {
        return accessor.greater(
            containerPtr,  
            index,
            other.containerPtr,
            other.index
        );
    }
    
    
    /**
     * @brief check whether the iterator is behind or equal a second one.
     * @return true if the iterator is behind or equal, false otherwise
     */
    template< bool T=isRandomAccessable>
    HDINLINE
    auto 
    operator<=(ReverseDeepIterator const & other)
    -> 
    typename std::enable_if<
        T == true, 
        bool
    >::type
    {

        return *this < other || *this == other;
    }

    
    /**
     * @brief check whether the iterator is ahead or equal a second one.
     * @return true if the iterator is ahead or equal, false otherwise
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto
    operator>=(ReverseDeepIterator const & other)
    -> 
    typename std::enable_if<
        T == true, 
        bool
    >::type
    {
        return *this > other || *this == other;
    }
    
    
    /**
     * @return get the element at the specified position.
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    auto
    operator[](IndexType const & index)
    -> 
    typename std::enable_if<
        T == true, 
        ComponentReference
    >::type
    {
        return accessor.get(
            containerPtr, 
            index
        );
    }
    
    
    /**
     * @brief check whether the iterator is after the last element
     * @return true, if it is, false if it is not after the last element
     */
    HDINLINE 
    auto 
    isAfterLast()
    const
    -> 
    bool
    {
        return navigator.isAfterLast(
            containerPtr, 
            index
        );
    }
    
    
    /**
     * @brief check whether the iterator is before the first element
     * @return true, if it is, false if it is not after the last element
     */
    HDINLINE 
    auto
    isBeforeFirst()
    const
    -> 
    bool
    {
        return navigator.isBeforeFirst(
            containerPtr, 
            index
        );
    }
    

    /**
     * @brief This function set the iterator to the first element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     */
    HDINLINE 
    auto
    setToBegin()
    -> 
    void
    {
        navigator.begin(
            containerPtr,
            index
        );
    }
    
    
    /**
     * @brief This function set the iterator to the first element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     */
    HDINLINE 
    auto 
    setToBegin(ContainerPtr con)
    -> 
    void
    {
        containerPtr = con;
        navigator.begin(
            containerPtr, 
            index
        );
    }
    
        
    /**
     * @brief This function set the iterator to the first element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     */
    HDINLINE 
    auto
    setToBegin(ContainerRef con)
    -> 
    void
    {
        containerPtr = &con;
        navigator.begin(
            containerPtr, 
            index
        );
    }
    
    
    /**
     * @brief This function set the iterator to the after-last-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     * */
    HDINLINE
    auto
    setToEnd(ContainerPtr con)
    -> 
    void
    {
        containerPtr = con;
        navigator.end(
            containerPtr,
            index
        );
    }
    
    
    /**
     * @brief This function set the iterator to the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     * */
    HDINLINE
    auto
    setToRend(ContainerPtr con)
    -> 
    void
    {
        containerPtr = con;
        navigator.rend(
            containerPtr,
            index
        );
    }

    
    /**
     * @brief This function set the iterator to the last element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     */
    HDINLINE
    auto
    setToRbegin(ContainerRef con)
    -> 
    void
    {
        containerPtr = &con;
        navigator.rbegin(
            containerPtr, 
            index
        );
    }
    
    
    /**
     * @brief This function set the iterator to the last element. This function
     * set also all childs to the begin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     */
    HDINLINE
    auto
    setToRbegin(ContainerPtr con)
    -> 
    void
    {
        containerPtr = con;
        navigator.rbegin(
            containerPtr,
            index
        );
    }
    
    
    /**
     * @brief This function set the iterator to the last element. This function
     * set also all childs to rbegin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     */
    HDINLINE
    auto
    setToRbegin()
    -> 
    void
    {
        navigator.rbegin(
            containerPtr, 
            index
        );
    }
    
        
    /**
     * @brief goto a successor element
     * @param jumpsize Distance to the successor element
     * @return The result value is importent if the iterator is in a middle 
     * layer. When we reach the end of the container, we need to give the higher 
     * layer informations about the remainig elements, we need to overjump. This 
     * distance is the return value of this function.
     */
    HDINLINE 
    auto 
    gotoNext(RangeType const & steps)
    ->
    RangeType
    {
        return navigator.next(
            containerPtr, 
            index, 
            steps
        );
    }
    
    
    /**
     * @brief goto a previos element
     * @param jumpsize Distance to the previous element
     * @return The result value is importent if the iterator is in a middle 
     * layer. When we reach the end of the container, we need to give the higher 
     * layer informations about the remainig elements, we need to overjump. This 
     * distance is the return value of this function.
     */
    HDINLINE
    auto 
    gotoPrevious(RangeType const & steps)
    ->
    RangeType
    {
        auto result = navigator.previous(
            containerPtr, 
            index, 
            steps
        );

        return result;
    }
    
    
    /**
     * @brief if the container has a constant size, this function can caluculate
     * it.
     * @return number of elements within the container. This include the child
     * elements
     */
    template<
        bool T = hasConstantSize>
    HDINLINE
    auto
    nbElements()
    const
    -> 
    typename std::enable_if<
        T == true,
        int32_t
    >::type
    {
        return navigator.size(containerPtr);
    }

    HDINLINE
    ContainerPtr
    getContainerPtr()
    const
    {
        return containerPtr;
    }
    
    HDINLINE
    IndexType const &
    getIndex()
    const
    {
        return index;
    }

private:
} ; // struct ReverseDeepIterator


namespace details 
{
    
    
/**
 * @brief This function is used in makeView. The function is a identity function
 * for deepiterator::NoChild
 */
template<
    typename TContainer,
    typename TChild,
// SFIANE Part
    typename TChildNoRef = typename std::decay<TChild>::type,
    typename = typename std::enable_if<
        std::is_same<
            TChildNoRef,
            deepiterator::NoChild
        >::value
    >::type
>
HDINLINE
auto
makeReverseIterator( TChild &&)
->
deepiterator::NoChild
{

    return deepiterator::NoChild();
}


/**
 * @brief bind an an iterator concept to an containertype. The concept has no 
 * child.
 * @tparam TContainer type of the container
 * @param concept an iterator prescription
 * 
 */
template<
    typename TContainer,
    typename TPrescription,
    typename TPrescriptionNoRef =typename std::decay<TPrescription>::type, 
    typename TContainerNoRef = typename std::decay<TContainer>::type, 
    typename ContainerCategoryType = typename traits::ContainerCategory<
        TContainerNoRef
    >::type,
    typename IndexType = typename deepiterator::traits::IndexType<
        TContainerNoRef,
        ContainerCategoryType
    >::type,
    bool isBidirectional = deepiterator::traits::IsBidirectional<
        TContainer, 
        ContainerCategoryType
    >::value,
    bool isRandomAccessable = deepiterator::traits::IsRandomAccessable<
        TContainer, 
        ContainerCategoryType
    >::value,
    bool hasConstantSize = traits::HasConstantSize<TContainer>::value,
    typename = typename std::enable_if<not std::is_same<
        TContainerNoRef, 
        deepiterator::NoChild
    >::value>::type
>
HDINLINE
auto 
makeReverseIterator (
    TPrescription && concept
)
->
ReverseDeepIterator<
        TContainer,
        decltype(makeAccessor<TContainer>(
            deepiterator::forward<TPrescription>(concept).accessor
        )),
        decltype(makeNavigator<TContainer>(
            deepiterator::forward<TPrescription>(concept).navigator
        )),
        decltype(makeReverseIterator<
            typename traits::ComponentType<TContainer>::type>(
                deepiterator::forward<TPrescription>(concept).child
            )
        ),
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable>
{
    using ContainerType = TContainer;
    using AccessorType =  decltype(
        makeAccessor<ContainerType>(
            deepiterator::forward<TPrescription>(concept).accessor
        )
    );
    using NavigatorType = decltype(
        makeNavigator<ContainerType>(
            deepiterator::forward<TPrescription>(concept).navigator
        )
    );
    using ChildType = decltype(
        makeReverseIterator<typename traits::ComponentType<TContainer>::type>(
            deepiterator::forward<TPrescription>(concept).child
        )
    );


    using Iterator = ReverseDeepIterator<
        ContainerType,
        AccessorType,
        NavigatorType,
        ChildType,
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable
    >;
 
    return Iterator(
        makeAccessor<ContainerType>(
            deepiterator::forward<TPrescription>(concept).accessor
        ),
        makeNavigator<ContainerType>(
            deepiterator::forward<TPrescription>(concept).navigator
        ),
        makeReverseIterator<typename traits::ComponentType<TContainer>::type>(
            deepiterator::forward<TPrescription>(concept).child
        )
    );
}

} // namespace details


/**
 * @brief Bind a container to a virtual iterator.  
 * @param con The container you like to inspect
 * @param iteratorPrescription A virtual iterator, which describes the behavior of 
 * the iterator
 * @return An Iterator. It is set to the first element.
 */
template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TAccessor,
    typename TNavigator,
    typename TChild,
    typename ContainerCategoryType = typename traits::ContainerCategory<
        TContainerNoRef
    >::type,
    typename IndexType = typename deepiterator::traits::IndexType<
        TContainerNoRef
    >::type,
    bool isBidirectional = deepiterator::traits::IsBidirectional<
        TContainerNoRef, 
        ContainerCategoryType
    >::value,
    bool isRandomAccessable = deepiterator::traits::IsRandomAccessable<
        TContainerNoRef, 
        ContainerCategoryType
    >::value,
    bool hasConstantSize = traits::HasConstantSize<
        TContainerNoRef
    >::value>
HDINLINE 
auto
makeReverseIterator(
    TContainer && container,
    deepiterator::details::IteratorPrescription<
        TAccessor,
        TNavigator,
        TChild> && concept)
-> 
ReverseDeepIterator<
        typename std::decay<TContainer>::type,
        decltype(details::makeAccessor(container, concept.accessor)),
        decltype(details::makeNavigator(container, concept.navigator)),
        decltype(details::makeReverseIterator<
            typename traits::ComponentType<
                typename std::decay<TContainer>::type
            >::type
        >(concept.childIterator)),
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable>         
{
    using ContainerType = typename std::decay<TContainer>::type;
    using ComponentType = typename traits::ComponentType<ContainerType>::type;
    
    using AccessorType = decltype(
        details::makeAccessor(
            container, 
            concept.accessor
        )
    );
    using NavigatorType = decltype(
        details::makeNavigator(
            container, 
            concept.navigator
        )
    );
    
    using ChildType = decltype(
        details::makeReverseIterator<ComponentType>(concept.childIterator)
    );
    

    using Iterator = ReverseDeepIterator<
        ContainerType,
        AccessorType,
        NavigatorType,
        ChildType,
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable
    >;
    
    return Iterator(
        container, 
        details::makeAccessor<ContainerType>(),
        details::makeNavigator<ContainerType>(concept.navigator),
        details::makeReverseIterator<ComponentType>(concept.childIterator));
}

} // namespace deepiterator

template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TChild,
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf>
std::ostream& operator<<(
    std::ostream& out, 
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf> const & iter
)
{
    out << "conPtr " << iter.containerPtr << " index " << iter.index << "Child: " << std::endl << iter.childIterator;
    return out;
}

template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf>
std::ostream& operator<<(
    std::ostream& out, 
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf> const & iter
)
{
    out << "conPtr " << iter.containerPtr << " index " << iter.index;
    return out;
}

/**
 * @brief this function compares the forward and the reverse iterator. You only can use it, if the template paremter are equal
 * 
 */
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf
>
HDINLINE 
auto
operator==(
    deepiterator::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & lhs,
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & rhs
)
-> 
bool
{
    return lhs.getContainerPtr() == rhs.getContainerPtr() && lhs.getIndex() == rhs.getIndex();
}

/**
 * @brief this function compares the forward and the reverse iterator. You only can use it, if the template paremter are equal
 * 
 */
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf
>
HDINLINE 
auto
operator==(
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & lhs,
    deepiterator::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & rhs
)
-> 
bool
{
    return rhs == lhs;
}

template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf
>
HDINLINE 
auto
operator!=(
    deepiterator::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & lhs,
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & rhs
)
-> 
bool
{
    return lhs.getContainerPtr() != rhs.getContainerPtr() || lhs.getIndex() != rhs.getIndex();
}

/**
 * @brief this function compares the forward and the reverse iterator. You only can use it, if the template paremter are equal
 * 
 */
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf
>
HDINLINE 
auto
operator!=(
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & lhs,
    deepiterator::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        deepiterator::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & rhs
)
-> 
bool
{
    return rhs != lhs;
}


/**
 * @brief this function compares the forward and the reverse iterator. You only can use it, if the template paremter are equal
 * 
 */
template<
    typename TContainer, 
    typename TAccessor, 
    typename TChild,
    typename TNavigator, 
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf
>
HDINLINE 
auto
operator==(
    deepiterator::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & lhs,
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & rhs
)
-> 
bool
{
    return lhs.getContainerPtr() == rhs.getContainerPtr() && lhs.getIndex() == rhs.getIndex() && rhs.getChild() == lhs.getChild();
}

/**
 * @brief this function compares the forward and the reverse iterator. You only can use it, if the template paremter are equal
 * 
 */
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TIndexType,
    typename TChild,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf
>
HDINLINE 
auto
operator==(
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & lhs,
    deepiterator::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & rhs
)
-> 
bool
{
    return rhs == lhs;
}

template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TIndexType,
    typename TChild,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf
>
HDINLINE 
auto
operator!=(
    deepiterator::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & lhs,
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & rhs
)
-> 
bool
{
    return not (lhs == rhs);
}

/**
 * @brief this function compares the forward and the reverse iterator. You only can use it, if the template paremter are equal
 * 
 */
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TIndexType,
    typename TChild,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf
>
HDINLINE 
auto
operator!=(
    deepiterator::ReverseDeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & lhs,
    deepiterator::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        TChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf
    > const & rhs
)
-> 
bool
{
    return rhs != lhs;
}
