

/**
 * \class DeepIterator
 * @author Sebastian Hahn (t.hahn@hzdr.de )
 * 
 * @brief The DeepIterator class is used to iterator over interleaved data 
 * structures. The simplest example for an interleaved data structure is 
 * std::vector< std::vector< int > >. The deepiterator iterates over all ints 
 * within the structure. 
 * 
 * Inside the deepiterator are two variables. These are importent for the 
 * templates. These Variables are:
 * 1. containerPtr: is the pointer to the container, given by the constructor
 * 2. index: the current element index within the container.
 * 
 * 
 * @tparam TContainer : This one describes the container, over whose elements 
 * you would like to iterate. This template need the trait \b ComponentType has 
 * a specialication for TContainer. This trait gives the type of the components 
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
   It it is recommended to use DeepIterator as TChild.
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
The DeepIterator supports up to four constructors: begin, end, rbegin, rend. To 
get the right one, we had four classes in details::constructorType. The 
constructors has five parameters: 
    ContainerPtr container, 
    TAccessor && accessor, 
    TNavigator && navigator,
    TChild && child,
    details::constructorType::__
If your container has no interleaved layer, you can use \b hzdr::NoChild as child.
A DeepIterator is bidirectional if the flag isBidirectionalSelf is set to true 
and all childs are bidirectional. The same applies to random accessablity.

We had two algorithms inside the DeepIterator. The first one is used to find the
first element within a nested data structure. The second one is used to find the
next element within the data structure. Lets start with the find the first 
element procedure. The setToBegin function has an optional parameter, this is 
the container over which the iterator walks. The first decision "Has Childs" is
done by the compiler. We had two different DeepIterators. One for the has childs
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
the deepiterator doesnt overjump elements. It walks step by step.
\image html images/setTobegin.png
 */
#include "deepiterator/iterator/categorie/ArrayNDLike.hpp"
#pragma once
#include "deepiterator/traits/Traits.hpp"
#include "deepiterator/definitions/forward.hpp"
#include "deepiterator/definitions/NoChild.hpp"
#include "deepiterator/definitions/hdinline.hpp"
#include "deepiterator/iterator/Accessor.hpp"
#include "deepiterator/iterator/Navigator.hpp"
#include "deepiterator/iterator/Prescription.hpp"
#include <limits>
#include <cassert>
#include <type_traits>
#include <sstream>
#include <typeinfo>

namespace hzdr 
{

namespace details 
{
namespace constructorType
{
struct begin{};
struct rbegin{};
struct end{};
struct rend{};
}
}

template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TChild,
    typename TIndexType,
    bool hasConstantSizeSelf = false,
    bool isBidirectionalSelf = true,
    bool isRandomAccessableSelf = true>
struct DeepIterator
{
// datatypes
    
public:
    typedef TContainer                                                  ContainerType;
    typedef ContainerType&                                              ContainerRef;
    typedef ContainerType*                                              ContainerPtr;
    
    typedef TAccessor                                                   Accessor;
    typedef TNavigator                                                  Navigator;
    
// child things
    typedef typename hzdr::traits::ComponentType<ContainerType>::type   ComponentType;
    typedef TChild                                                  ChildIterator;
    typedef typename ChildIterator::ReturnType                      ReturnType;

// container stuff
    typedef TIndexType   IndexType;
    
// protected:
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
    HDINLINE DeepIterator() = default;
    HDINLINE DeepIterator(DeepIterator const &) = default;
    HDINLINE DeepIterator(DeepIterator &&) = default;
    
    /**
     * @brief This constructor is used to create a iterator in a middle layer. 
     * The container must be set with setToBegin or setToRbegin.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child iterator for the next layer
     */
    template<
        typename TAccessor_, 
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    DeepIterator(TAccessor_ && accessor, 
                 TNavigator_ && navigator,
                 TChild_ && child
                ):
        containerPtr(nullptr),
        index(static_cast<IndexType>(0)),
        childIterator(hzdr::forward<TChild_>(child)),
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor))
        
    {}
    
        /**
     * @brief This constructor is used to create an iterator at the begin element.
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child iterator for the next layer
     * @param details::constructorType::begin used to specify that the begin 
     * element is needed. We recommend details::constructorType::begin() as 
     * parameter.
     */
    template<
        typename TAccessor_,
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    DeepIterator(
                ContainerPtr container, 
                TAccessor_&& accessor, 
                TNavigator_&& navigator,
                TChild_&& child,
                details::constructorType::begin 
                ):
        containerPtr(container),
        index(static_cast<IndexType>(0)),
        childIterator(hzdr::forward<TChild_>(child)),
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor))
        
    {
         setToBegin(container);
    }
    
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
    DeepIterator(
                ContainerPtr container, 
                TAccessor_&& accessor, 
                TNavigator_&& navigator,
                TChild_&& child,
                details::constructorType::rbegin
                ):
        containerPtr(container),
        index(static_cast<IndexType>(0)),
        childIterator(hzdr::forward<TChild_>(child)),
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor))
        
    {
        setToRbegin(container);
    }
    
    /**
     * @brief This constructor is used to create an iterator at the end element.
     * @param ContainerPtr A pointer to the container you like to iterate through.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child iterator for the next layer
     * @param details::constructorType::end used to specify that the end
     * element is needed. We recommend details::constructorType::end() as 
     * parameter.
     */
    template<
        typename TAccessor_,
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    DeepIterator(
                ContainerPtr container, 
                TAccessor_&& accessor, 
                TNavigator_&& navigator,
                TChild_&& child,
                details::constructorType::end
                ):
        containerPtr(container),
        index(static_cast<IndexType>(0)),
        childIterator(hzdr::forward<TChild_>(child)),
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor))
        
    {
        setToEnd(container);
    }
    
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
    DeepIterator(
                ContainerPtr container, 
                TAccessor_&& accessor, 
                TNavigator_&& navigator,
                TChild_&& child,
                details::constructorType::rend
                ):
        containerPtr(container),
        index(static_cast<IndexType>(0)),
        childIterator(hzdr::forward<TChild_>(child)),
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor))
        
    {
        setToRend(container);
    }
    
    HDINLINE DeepIterator& operator=(DeepIterator const &) = default;
    HDINLINE DeepIterator& operator=(DeepIterator &&) = default;
    
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
     * @brief compares the DeepIterator with an other DeepIterator.
     * @return true: if the iterators are at different positions, false
     * if they are at the same position
     */
    HDINLINE
    bool
    operator!=(const DeepIterator& other)
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
     * @brief compares the DeepIterator with an other DeepIterator.
     * @return false: if the iterators are at different positions, true
     * if they are at the same position
     */
    HDINLINE
    bool
    operator==(const DeepIterator& other)
    const
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
    DeepIterator&
    operator++()
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
    DeepIterator
    operator++(int)
    {
        DeepIterator tmp(*this);
        if(isBeforeFirst())
        {
            setToBegin();
            return tmp;
        }
        gotoNext(1u);
        return tmp;
    }
    

    /**
     * @brief goto the previous element. If the iterator is at after-first-element,
     * it is set to the rbegin element. The iterator need to be bidirectional to
     * support this function.
     * @return reference to the previous element
     */
    template<
        bool T = isBidirectional>
    HDINLINE
    typename std::enable_if<T == true, DeepIterator>::type
    operator--()
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
    template<
        bool T = isBidirectional>
    HDINLINE
    typename std::enable_if<T == true, DeepIterator>::type
    operator--(int)
    {
        DeepIterator tmp(*this);
        
        --(*this);
        return tmp;
    }

    /**
     * @brief creates an iterator which is jumpsize elements ahead. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     * @return iterator which is jumpsize elements ahead
     */
    template<
        bool T = isRandomAccessable>    
    HDINLINE 
    typename std::enable_if<T == true, DeepIterator>::type
    operator+(uint const & jumpsize)
    {
        DeepIterator tmp(*this);
        tmp += jumpsize;
        return tmp;
    }
    
        /**
     * @brief set the iterator jumpsize elements ahead. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     */
    template<
        bool T = isRandomAccessable>    
    HDINLINE 
    typename std::enable_if<T == true, DeepIterator&>::type
    operator+=(uint const & jumpsize)
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
    template< 
        bool T = hasConstantSizeChild> 
    HDINLINE 
    typename std::enable_if<T == true, uint>::type 
    gotoNext(uint const & jumpsize)
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
            childIterator.setToBegin(accessor.get(containerPtr, index));
            childIterator += childJumps;
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
    template< 
        bool T = hasConstantSizeChild> 
    HDINLINE 
    typename std::enable_if<T == false, uint>::type 
    gotoNext(uint const & jumpsize)
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
     * @brief creates an iterator which is jumpsize elements behind. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the previous element
     * @return iterator which is jumpsize elements behind
     */
    template<
        bool T = isRandomAccessable>    
    HDINLINE 
    typename std::enable_if<T == true, DeepIterator>::type
    operator-(uint const & jumpsize)
    {
        DeepIterator tmp(*this);
        tmp -= jumpsize;
        return tmp;
    }
    /**
     * @brief set the iterator jumpsize elements behind. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     */
    template<bool T = isRandomAccessable>
    HDINLINE 
    typename std::enable_if<T == true, DeepIterator&>::type
    operator-=(const uint & jumpsize)
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
    typename std::enable_if<T == true, uint>::type    
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
                childIterator -= childJumps;

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
    template< 
        bool T = hasConstantSizeChild> 
    HDINLINE 
    typename std::enable_if<T == false, uint>::type 
    gotoPrevious(uint const & jumpsize)
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
                navigator.previous(containerPtr, index, 1u);
                if(not isBeforeFirst())
                    childIterator.setToRbegin(accessor.get(containerPtr, index));
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
    typename std::enable_if<T == true, bool>::type
    operator<(DeepIterator const & other)
    {
        if(accessor.lesser(
            containerPtr, 
            index,
            other.containerPtr,
            other.index))
           return true;
        if(accessor.equal(containerPtr, 
            index,
            other.containerPtr,
            other.index) &&
           childIterator < other.childIterator)
            return true;
        return false;
    }
    
    /**
     * @brief check whether the iterator is ahead a second one.
     * @return true if the iterator is ahead, false otherwise
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T == true, bool>::type
    operator>(DeepIterator const & other)
    {
        if(accessor.greater(
            containerPtr, 
            index,
            other.containerPtr,
            other.index))
           return true;
        if(accessor.equal(containerPtr, 
            index,
            other.containerPtr,
            other.index) &&
           childIterator > other.childIterator)
            return true;
        return false;
    }
    
            /**
     * @brief check whether the iterator is behind or equal a second one.
     * @return true if the iterator is behind or equal, false otherwise
     */
    template< bool T=isRandomAccessable>
    HDINLINE
    typename std::enable_if<T == true, bool>::type
    operator<=(DeepIterator const & other)
    {
        return *this < other || *this == other;
    }
    
        /**
     * @brief check whether the iterator is ahead or equal a second one.
     * @return true if the iterator is ahead or equal, false otherwise
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T == true, bool>::type
    operator>=(DeepIterator const & other)
    {
        return *this > other || *this == other;
    }
    
    /**
     * @return get the element at the specified position.
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T == true, ReturnType&>::type
    operator[](IndexType const & index)
    {
        DeepIterator tmp(*this);
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
    void
    setToBegin()
    {
        navigator.begin(containerPtr, index);
        // check whether the iterator is at a valid element
        while(not isAfterLast())
        {
            childIterator.setToBegin((accessor.get(
                containerPtr, index
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
    void
    setToBegin(TContainer& con)
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
    void
    setToBegin(TContainer* ptr)
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
    void
    setToEnd(TContainer* ptr)
    {
        containerPtr = ptr;
        navigator.end(containerPtr, index);
    }
    
    /**
     * @brief This function set the iterator to the before-first-element. 
     * @param TContainer The container over which the iterator walks is changed 
     * to TContainer.
     * */
    HDINLINE
    void
    setToRend(TContainer* ptr)
    {
        containerPtr = ptr;
        navigator.rend(containerPtr, index);
    }

        /**
     * @brief This function set the iterator to the last element. This function
     * set also all childs to rbegin. If the container hasnt enough elements
     * it should be set to the after-last-element or the before-first-element. 
     */
    HDINLINE
    void 
    setToRbegin()
    {
        navigator.rbegin(containerPtr, index);
        // check whether the iterator is at a valid element
//         if(isBeforeFirst())
//         {
//             std::stringstream str;
//             str << "index " << index << "( " << hzdr::detail::idxndToInt<3>(index, containerPtr->dim()) << " ) is before first " << std::endl;
//         
//             std::cout << str.str() << std::endl;
//         }
        while(not isBeforeFirst())
        {
            childIterator.setToRbegin((accessor.get(containerPtr, index)));
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
    void 
    setToRbegin(ContainerRef con)
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
    void 
    setToRbegin(ContainerPtr ptr)
    {
        containerPtr = ptr;
        setToRbegin();
    }

    /**
     * @brief check whether the iterator is after the last element
     * @return true, if it is, false if it is not after the last element
     */
    HDINLINE 
    bool
    isAfterLast()
    const
    {
        return navigator.isAfterLast(containerPtr, index);
    }
    
    /**
     * @brief check whether the iterator is before the first element
     * @return true, if it is, false if it is not after the last element
     */
    HDINLINE 
    bool
    isBeforeFirst()
    const
    {
        return navigator.isBeforeFirst(containerPtr, index);
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
    typename std::enable_if<T == true, int>::type
    nbElements()
    const
    {
        return childIterator.nbElements() * navigator.size(containerPtr);
    }
} ; // struct DeepIterator





/** ************************************+
 * @brief The flat implementation. This deepiterator has no childs. 
 * ************************************/
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator,
    typename TIndexType,
    bool hasConstantSizeSelf,
    bool isBidirectionalSelf,
    bool isRandomAccessableSelf>
struct DeepIterator<
    TContainer,     
    TAccessor, 
    TNavigator,
    hzdr::NoChild,
    TIndexType,
    hasConstantSizeSelf,
    isBidirectionalSelf,
    isRandomAccessableSelf>
{
// datatypes
public:
    typedef TContainer                                                  ContainerType;
    typedef ContainerType*                                              ContainerPtr;
    typedef ContainerType&                                              ContainerReference;
    
    typedef typename hzdr::traits::ComponentType<ContainerType>::type   ComponentType;
    typedef ComponentType*                                              ComponentPtr;
    typedef ComponentType&                                              ComponentReference;
    typedef ComponentReference                                          ReturnType;
    typedef TAccessor                                                   Accessor;
    typedef TNavigator                                                  Navigator;
    

    typedef TIndexType   IndexType;

// Variables    
// protected:   
    Navigator navigator;
    Accessor accessor;
    hzdr::NoChild childIterator;
    

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


    HDINLINE DeepIterator() = default;
    HDINLINE DeepIterator(const DeepIterator&) = default;
    HDINLINE DeepIterator(DeepIterator&&) = default;

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
    DeepIterator(
                ContainerPtr container, 
                TAccessor_&& accessor, 
                TNavigator_&& navigator,
                TChild_ const &,
                details::constructorType::begin 
                ):
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor)),
        childIterator(),
        containerPtr(container),
        index(0)

        
    {
        setToBegin(container);
    }
    
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
    DeepIterator(
                ContainerPtr container, 
                TAccessor_&& accessor, 
                TNavigator_&& navigator,
                 TChild_ const &,
                details::constructorType::rbegin
                ):
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor)),
        childIterator(),
        containerPtr(container),
        index(0)
    {
        setToRbegin(container);
    }
    
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
        typename TChild_>
    HDINLINE
    DeepIterator(
                ContainerPtr container, 
                TAccessor_&& accessor, 
                TNavigator_&& navigator,
                 TChild_ const &,
                details::constructorType::end
                ):
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor)),
        containerPtr(container),
        index(0)
    {
        setToEnd(container);
    }
    
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
    DeepIterator(
                ContainerPtr container, 
                TAccessor_&& accessor, 
                TNavigator_&& navigator,
                 TChild_ const &,
                details::constructorType::rend
                ):
        
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor)),
        containerPtr(container),
        index(0)
        
    {
        setToRend(container);
    }
    
    HDINLINE DeepIterator& operator=(DeepIterator const &) = default;
    HDINLINE DeepIterator& operator=(DeepIterator &&) = default;
    /**
     * @brief This constructor is used to create a iterator in a middle layer. 
     * The container must be set with setToBegin or setToRbegin.
     * @param accessor The accessor
     * @param navigator The navigator, needs a specified offset and a jumpsize.
     * @param child use hzdr::NoChild()
     */
    template<
        typename TAccessor_, 
        typename TNavigator_>
    HDINLINE
    DeepIterator(TAccessor_ && accessor, 
                 TNavigator_ && navigator,
                 hzdr::NoChild const &
                ):
        
        navigator(hzdr::forward<TNavigator_>(navigator)),
        accessor(hzdr::forward<TAccessor_>(accessor)),
        containerPtr(nullptr),
        index(0)
    {}
    
    
    
    /**
     * @brief goto the next element. If the iterator is at the before-first-element
     * it is set to the begin element.
     * @return reference to the next element
     */
    HDINLINE
    DeepIterator&
    operator++()
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
     * @brief goto the next element. If the iterator is at the before-first-element
     * it is set to the begin element.
     * @return reference to the current element
     */
    HDINLINE
    DeepIterator
    operator++(int)
    {
        DeepIterator tmp(*this);
        navigator.next(
            containerPtr, 
            index, 
            1u
        );
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
     * @brief compares the DeepIterator with an other DeepIterator.
     * @return true: if the iterators are at different positions, false
     * if they are at the same position
     */
    HDINLINE
    bool
    operator!=(const DeepIterator& other)
    const
    {
        
        return (containerPtr != other.containerPtr
             || index != other.index)
             && (not isAfterLast() || not other.isAfterLast())
             && (not isBeforeFirst() || not other.isBeforeFirst());
    }
    
    /**
     * @brief compares the DeepIterator with an other DeepIterator.
     * @return false: if the iterators are at different positions, true
     * if they are at the same position
     */
    HDINLINE
    bool
    operator==(const DeepIterator& other)
    const
    {
        return not (*this != other);
    }
    
    /**
     * @brief goto the previous element. If the iterator is at after-first-element,
     * it is set to the rbegin element. The iterator need to be bidirectional to
     * support this function.
     * @return reference to the previous element
     */
    template<bool T = isBidirectional>
    HDINLINE 
    typename std::enable_if<T, DeepIterator& >::type
    operator--()
    {
        navigator.previous(
            containerPtr, 
            index, 
            1u
        );
        return *this;
    }
    
    /**
     * @brief goto the previous element. If the iterator is at after-first-element,
     * it is set to the rbegin element. The iterator need to be bidirectional to
     * support this function.
     * @return reference to the current element
     */
    template<bool T = isBidirectional>
    HDINLINE 
    typename std::enable_if<T, DeepIterator >::type
    operator--(int)
    {
        DeepIterator tmp(*this);
        navigator.previous(
            containerPtr, 
            index, 
            1u
        );
        return tmp;
    }
    
     /**
     * @brief set the iterator jumpsize elements ahead. The iterator 
     * need to be random access to support this function.
     * @param jumpsize distance to the next element
     */
    template<bool T=isRandomAccessable>
    HDINLINE
    typename std::enable_if<T == true, DeepIterator&>::type
    operator+=(RangeType const & jump)
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
    typename std::enable_if<T == true, DeepIterator&>::type
    operator-=(RangeType const & jump)
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
    typename std::enable_if<T == true, DeepIterator>::type
    operator+(RangeType const & jump)
    {
        DeepIterator tmp = *this;
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
    typename std::enable_if<T == true, DeepIterator>::type
    operator-(RangeType const & jump)
    {
        DeepIterator tmp = *this;
        tmp-=jump;
        return tmp;
    }
    
        /**
     * @brief check whether the iterator is behind a second one.
     * @return true if the iterator is behind, false otherwise
     */
    template< bool T=isRandomAccessable>
    HDINLINE
    typename std::enable_if<T == true, bool>::type
    operator<(DeepIterator const & other)
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
    typename std::enable_if<T == true, bool>::type
    operator>(DeepIterator const & other)
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
    typename std::enable_if<T == true, bool>::type
    operator<=(DeepIterator const & other)
    {

        return *this < other || *this == other;
    }
    
            /**
     * @brief check whether the iterator is ahead or equal a second one.
     * @return true if the iterator is ahead or equal, false otherwise
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T == true, bool>::type
    operator>=(DeepIterator const & other)
    {
        return *this > other || *this == other;
    }
    
    /**
     * @return get the element at the specified position.
     */
    template<bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T == true, ComponentReference>::type
    operator[](IndexType const & index)
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
    bool
    isAfterLast()
    const
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
    bool
    isBeforeFirst()
    const
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
    void 
    setToBegin()
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
    void 
    setToBegin(ContainerPtr con)
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
    void 
    setToBegin(ContainerReference con)
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
    void 
    setToEnd(ContainerPtr con)
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
    void 
    setToRend(ContainerPtr con)
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
    void 
    setToRbegin(ContainerReference con)
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
    void 
    setToRbegin(ContainerPtr con)
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
    void 
    setToRbegin()
    {
        navigator.rbegin(
            containerPtr, 
            index
        );
    }
    
           /**
     * @brief goto a successor element
     * @param jumpsize Distance to the successor element
     * @return The result value is importent if the iterator is in a middle layer.
     * When we reach the end of the container, we need to give the higher layer
     * informations about the remainig elements, we need to overjump. This distance
     * is the return value of this function.
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
     * @return The result value is importent if the iterator is in a middle layer.
     * When we reach the end of the container, we need to give the higher layer
     * informations about the remainig elements, we need to overjump. This distance
     * is the return value of this function.
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
    typename std::enable_if<T, int>::type
    nbElements()
    const
    {
        return navigator.size(containerPtr);
    }


private:
} ; // struct DeepIterator


namespace details 
{
/**
 * @brief This function is used in makeView. The function is a identity function
 * for hzdr::NoChild
 */
#include <typeinfo>
template<
    typename TContainer,
    typename TChild,
// SFIANE Part
    typename TChildNoRef = typename std::decay<TChild>::type,
    typename = typename std::enable_if<std::is_same<TChildNoRef, hzdr::NoChild>::value>::type
>
HDINLINE
auto
makeIterator( TChild &&)
->
hzdr::NoChild
{

    return hzdr::NoChild();
}



/**
 * @brief bind an an iterator concept to an containertype. The concept has no child.
 * @tparam TContainer type of the container
 * @param concept an iterator concept
 * 
 */
template<
    typename TContainer,
    typename TPrescription,
    typename TPrescriptionNoRef =typename std::decay<TPrescription>::type, 
    typename TContainerNoRef = typename std::decay<TContainer>::type, 
    typename ContainerCategoryType = typename traits::ContainerCategory<TContainerNoRef>::type,
    typename IndexType = typename hzdr::traits::IndexType<
        TContainerNoRef,
        ContainerCategoryType
    >::type,
    bool isBidirectional = hzdr::traits::IsBidirectional<TContainer, ContainerCategoryType>::value,
    bool isRandomAccessable = hzdr::traits::IsRandomAccessable<TContainer, ContainerCategoryType>::value,
    bool hasConstantSize = traits::HasConstantSize<TContainer>::value,
    typename = typename std::enable_if<not std::is_same<TContainerNoRef, hzdr::NoChild>::value>::type
>
HDINLINE
auto 
makeIterator (
    TPrescription && concept
)
->
DeepIterator<
        TContainer,
        decltype(makeAccessor<TContainer>(hzdr::forward<TPrescription>(concept).accessor)),
        decltype(makeNavigator<TContainer>(hzdr::forward<TPrescription>(concept).navigator)),
        decltype(makeIterator<
            typename traits::ComponentType<TContainer>::type>(hzdr::forward<TPrescription>(concept).child)),
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable>
{
    typedef TContainer                                          ContainerType;

    typedef decltype(makeAccessor<ContainerType>(hzdr::forward<TPrescription>(concept).accessor))      AccessorType;
    typedef decltype(makeNavigator<ContainerType>(hzdr::forward<TPrescription>(concept).navigator))    NavigatorType;
    typedef decltype(makeIterator<
            typename traits::ComponentType<TContainer>::type>(hzdr::forward<TPrescription>(concept).child)) ChildType;


    typedef DeepIterator<
        ContainerType,
        AccessorType,
        NavigatorType,
        ChildType,
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable>         Iterator;
 
    return Iterator(
        makeAccessor<ContainerType>(hzdr::forward<TPrescription>(concept).accessor),
        makeNavigator<ContainerType>(hzdr::forward<TPrescription>(concept).navigator),
        makeIterator<typename traits::ComponentType<TContainer>::type>(hzdr::forward<TPrescription>(concept).child));;
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
    typename ContainerCategoryType = typename traits::ContainerCategory<TContainerNoRef>::type,
    typename IndexType = typename hzdr::traits::IndexType<TContainerNoRef>::type,
    bool isBidirectional = hzdr::traits::IsBidirectional<TContainerNoRef, ContainerCategoryType>::value,
    bool isRandomAccessable = hzdr::traits::IsRandomAccessable<TContainerNoRef, ContainerCategoryType>::value,
    bool hasConstantSize = traits::HasConstantSize<TContainerNoRef>::value>
HDINLINE 
auto
makeIterator(
    TContainer && container,
    hzdr::details::IteratorPrescription<
        TAccessor,
        TNavigator,
        TChild> && concept)
-> 
DeepIterator<
        typename std::decay<TContainer>::type,
        decltype(details::makeAccessor(container, concept.accessor)),
        decltype(details::makeNavigator(container, concept.navigator)),
        decltype(details::makeIterator<
            typename traits::ComponentType<
                typename std::decay<TContainer>::type>::type>(concept.childIterator)),
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable>         
{
    typedef typename std::decay<TContainer>::type               ContainerType;
    typedef typename traits::ComponentType<ContainerType>::type ComponentType;
    
    typedef decltype(details::makeAccessor(container, concept.accessor))      AccessorType;
    typedef decltype(details::makeNavigator(container, concept.navigator))    NavigatorType;
    typedef decltype(details::makeIterator<ComponentType>(concept.childIterator)) ChildType;
    

    typedef DeepIterator<
        ContainerType,
        AccessorType,
        NavigatorType,
        ChildType,
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable
        >         Iterator;
    
    return Iterator(
        container, 
        details::makeAccessor<ContainerType>(),
        details::makeNavigator<ContainerType>(concept.navigator),
        details::makeIterator<ComponentType>(concept.childIterator));
}

} // namespace hzdr

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
    hzdr::DeepIterator<
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
    hzdr::DeepIterator<
        TContainer,
        TAccessor,
        TNavigator,
        hzdr::NoChild,
        TIndexType,
        hasConstantSizeSelf,
        isBidirectionalSelf,
        isRandomAccessableSelf> const & iter
)
{
    out << "conPtr " << iter.containerPtr << " index " << iter.index;
    return out;
    
}
