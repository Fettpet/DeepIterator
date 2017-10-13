
/**
 * \class DeepIterator
 * @author Sebastian Hahn (t.hahn@hzdr.de )
 * 
 * @brief The DeepIterator class is used to iterator over interleaved data 
 * structures. The simplest example for an interleaved data structure is 
 * std::vector< std::vector< int > >. The deepiterator iterates over all ints 
 * within the structure. 
 * 
 * Inside the deepiterator are three variables. These are
 * importent for the templates. These Variables are:
 * 1. componentPtr: represent is a pointer to the current component
 * 2. containerPtr: is the pointer to the container, given by the constructor
 * 3. index: the current element index within the container
 * 
 * 
 * @tparam TContainer : This one describes the container, over whose elements you 
 * would like to iterate. This templeate has some conditions: I. The trait 
 * IsIndexable need a specialication for TContainer. This traits, says wheter 
 * TContainer is array like (has []-operator overloaded) or list like; II. The
 * trait ComponentType has a specialication for TContainer. This trait gives the type
 * of the components of TContainer; III. The function \b NeedRuntimeSize<TContainer>
 * need to be specified. For more details see NeedRuntimeSize.hpp ComponentType.hpp IsIndexable.hpp
 * @tparam TAccessor The accessor descripe the access to the components of TContainer.
 * The Accessor need a get Function: 
 * static TComponent* accessor.get(TContainer* , 
                                    TComponent*, 
                                    const TIndex&,
                                    const RuntimeVariables&)
 * The function get returns a pointer to the current component. We have 
 * implementeted an Accessor.
   @tparam TNavigator The navigator describe the way to walk through the data. This
 * policy need three functions specified. The first function gives an entry
   point in the container:
 * static void first(TContainer* conPtrIn, 
                     TContainer*& conPtrOut, 
                     TComponent*& compontPtrOut,
                     TIndex& indexOut, 
                     const TOffset& offset)
 * The function has two input parameter (conPtrIn and offset), the first is a pointer 
 * to the container given by the constructor and the second is the number of elements
 which are overjump by the navigator.
 * The other three paramter are output parameter. They are not given to the 
 * Accessor, so be sure, there are no conflict. 
 * The second function is the next function. These function goes to the next element
 * within the container:
   static void next(TContainer*, 
                    TComponent* elem, 
                    TIndex& index,  
                    const TRuntimeVariables& run);
 * The parameters are described above.
 * The third function decided, whether the end is reached. This function results
 * in true if the element is invalid and there are no reasons, that other threads
 in the same warp have valid elements. In other cases this function returns false.
 The structure of this function is:
    static
    bool 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const compontPtr,
          const TIndex& index, 
          const TRuntimeVariables& run);
 We have implemented a navigator. For more details Navigator
 @tparam TWrapper The wrapper decides whether a component is valid, or not. An 
 object is valid if it is part of the container. Otherwise it is not valid. This
 is importent for collective mutlithread programs. Some threads could have an 
 valid component, others havnt. Since all threads are collectiv, it could end in
 a dead look, since some threads are in the loop, others are not. A wrapper has 
 a constructor, an overloaded bool operator and an overloaded * operator. We 
 begin with the constructor. The constructor has five parameter: The first is 
 the result of the accessor.get. The other four are the variables of the deep -
 iterator:
 Wrapper(accessor.get(containerPtr, index, runtimeVariables),
         containerPtr, 
         componentPtr, 
         index, 
         runtimeVariables)
 The second function, the operator bool, is needed to check whether the value is
 valid. The operator* gives the elemnt. For more details see Wrapper
 @tparam TCollective The collective policy does the stuff for parallel execution
 of the Iterator. It defines an offset, a jumpsize and a synchronication function.
 The offset is the distance from the first element within the container to the 
 first element of the iterator. The jumpsize is the distance between two elements
 of the iterator.
 The functions have the headers:    
 void sync(); // the function for synchronication
 uint_fast32_t offset() const; // the offset
 uint_fast32_t jumpsize() const; // the jumpsize
 
 @tparam TChild The child is the template parameter to realize nested structures. 
 This template has several requirements: 
    1. it need to spezify an Iterator type. These type need operator++,  operator*,
        operator=, operator!= and a default constructor.
    2. it need an WrapperType type
    3. it need a begin and a end function. The result of the begin function must
       have the same type as the operator= of the iterator. The result of the 
       end function must have the same type as the operator!= of the iterator.
    4. default constructor
    5. copy constructor
    6. constructor with childtype and containertype as variables
    7. refresh(componentType*): for nested datastructures we start to iterate in
    deeper layers. After the end is reached, in this layers, we need to go to the
    next element in the current layer. Therefore we had an new component. This 
    component is given to the child.
 To use the Child we recommed the View.
 # Usage {#sectionD2}
 The first step to use the iterator is to define it. The template parameter are
 described above. After that you construct an instant of the iterator. To do this
 there are two constructors, one if you had a child and the second if you doesn't 
 have:
     DeepIterator(ContainerType* _ptr, 
                 const TRuntimeVariables& runtimeVariables); // no child
     DeepIterator(ContainerType* _ptr, 
                 const TRuntimeVariables& runtimeVariables,
                 TChild child); // with child
 To walk through the data the iterator has the ++ operator overload. It use the 
 next function from the navigator.
 The deepIterator serves two ways to check whether the end is reached or not:
 1. operator!=(nullptr_t): if you compare your iterator with a nullptr, the iterator
 use the isEnd function of the navigator, to decide, whether the end is reached,
 or not.
 2. operator!=(const deepIterator&): we compare the values of the variables 
 componenttype, index and containertype. If all of these are equal to the other
 instance of the iterator, the end is reached.
 \see Wrapper \see View \see Navigator \see Accessor \see Collectivity 
 \see IsIndexable \see NumberElements \see MaxElements
 */
#include <sstream>
#pragma once
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "PIC/Supercell.hpp"

#include "Iterator/Concept.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Collective.hpp"
#include "Iterator/Wrapper.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include <limits>
#include <cassert>
#include <type_traits>
#include "Traits/NumberElements.hpp"
#include "Traits/Componenttype.hpp"
#include "Definitions/hdinline.hpp"
#include <typeinfo>
#include "Traits/IsBidirectional.hpp"
#include "Traits/IsRandomAccessable.hpp"
#include "Traits/ContainerCategory.hpp"
#include "Traits/MaxElements.hpp"
#include "Traits/HasConstantSize.hpp"
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

/**
 * @tparam TContainer is the type of the container 
 * @tparam TChild ist ein virtueller Container oder NoChild
 */
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator, 
    typename TChild>
struct DeepIterator
{
// datatypes
    
public:
    typedef TContainer                                                  ContainerType;
    typedef ContainerType&                                              ContainerRef;
    typedef ContainerType*                                              ContainerPtr;
    
    typedef typename hzdr::traits::ComponentType<ContainerType>::type   ComponentType;
    typedef ComponentType*                                              ComponentPointer;
    typedef ComponentType&                                              ComponentReference;
    
    typedef TAccessor                                                   Accessor;
    typedef TNavigator                                                  Navigator;
    
// child things
    typedef TChild                                                  ChildIterator;
    typedef typename ChildIterator::ReturnType                      ReturnType;

// container stuff
    typedef typename traits::ContainerCategory<TContainer>::type    ContainerCategoryType;
    typedef typename traits::IndexType<ContainerCategoryType>::type   IndexType;
    
    static const bool isBidirectional = ChildIterator::isBidirectional && hzdr::traits::IsBidirectional<ContainerType>::value;
    static const bool isRandomAccessable = ChildIterator::isRandomAccessable && hzdr::traits::IsRandomAccessable<ContainerType>::value;
    
    static const bool hasConstantSizeChild = ChildIterator::hasConstantSize;

    
    static const bool hasConstantSize = traits::HasConstantSize<ContainerType>::value && hasConstantSizeChild;
    static const int_fast32_t selfCompileTimeSize = traits::MaxElements<ContainerType>::value;

    
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */
    HDINLINE DeepIterator() = default;
    HDINLINE DeepIterator(DeepIterator const &) = default;
    HDINLINE DeepIterator(DeepIterator &&) = default;
    
    template<
        typename TAccessor_, 
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    DeepIterator(TAccessor_ && accessor, 
                 TNavigator_ && navigator,
                 TChild_ && child
                ):
        childIterator(std::forward<TChild_>(child)),
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor))
        
    {}
    
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
        childIterator(std::forward<TChild_>(child)),
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor))
        
    {
        setToBegin(container);
    }
    
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
        childIterator(std::forward<TChild_>(child)),
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor))
        
    {
        setToRbegin(container);
    }
    
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
        childIterator(std::forward<TChild_>(child)),
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor))
        
    {
        setToEnd(container);
    }
    
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
        childIterator(std::forward<TChild_>(child)),
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor))
        
    {
        setToRend(container);
    }
    
    
    HDINLINE 
    DeepIterator(
        ContainerPtr containerPtr,
        IndexType && index,
        Accessor&& accessor, 
        Navigator&& navigator,
        ChildIterator&& child):
            containerPtr(containerPtr),
            index(std::forward<IndexType>(index)),
            childIterator(std::forward<ChildIterator>(child)),
            navigator(std::forward<Navigator>(navigator)),
            accessor(std::forward<Accessor>(accessor))
    {}

    
    
    HDINLINE
    auto
    operator*()
    ->
    ReturnType
    {
        return *childIterator;
    }
    
    
    HDINLINE
    bool
    operator!=(const DeepIterator& other)
    const
    {
        
        return not (*this == other);
    }
    
    HDINLINE 
    ReturnType
    operator->()
    {
        return *childIterator;
    }

    HDINLINE
    bool
    operator==(const DeepIterator& other)
    const
    {
//        std::cout << std::boolalpha << other.isBeforeFirst() << " && " << isBeforeFirst() << " index " << index << " " << other.index << std::endl;
        return (isAfterLast() && other.isAfterLast())
            || (isBeforeFirst() && other.isBeforeFirst())
            ||(containerPtr == other.containerPtr
            && index == other.index 
            && other.childIterator == childIterator);
    }
    
    /**
     * @brief goto the next element
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
    

    HDINLINE
    DeepIterator
    operator--()
    {
        if(isAfterLast())
        {
            setToRbegin();
            return *this;
        }
        gotoPrevious(1u);
        return *this;
    }
    
    template<
        bool T = isBidirectional>
    HDINLINE
    typename std::enable_if<T, DeepIterator>::type
    operator--(int)
    {
        DeepIterator tmp(*this);
        
        if(isAfterLast())
        {
            setToRbegin();
            return tmp;
        }
        gotoPrevious(1u);
        return tmp;
    }

    template<
        bool T = isRandomAccessable>    
    HDINLINE 
    DeepIterator
    operator+(typename std::enable_if<T  == true, int>::type jumpsize)
    {
        DeepIterator tmp(*this);
        tmp += jumpsize;
        return tmp;
    }
    

    HDINLINE 
    DeepIterator&
    operator+=(uint const & jumpsize)
    {
        auto tmpJump = jumpsize;
        if(isBeforeFirst())
        {
            --tmpJump;
            setToBegin();
        }
        gotoNext(jumpsize);
        return *this;

    }
    
    template< 
        bool T = hasConstantSizeChild> 
    HDINLINE 
    typename std::enable_if<T == true, uint>::type 
    gotoNext(uint const & jumpsize)
    {
        /**
         * The variable jumpsize is component from three other variables:
         * 1. The distance of the child to these end
         * 2. the number of childs we can overjump
         * 3. the remaining jumpsize for the new child
         */
        
        auto && remaining = childIterator.gotoNext(jumpsize);
        if(childIterator.isAfterLast() and remaining == 0)
        {
                remaining = jumpsize;
        }
        
        auto && childNbElements = childIterator.nbElements();
        auto && overjump = (remaining + childNbElements - 1) / childNbElements;
        int childJumps = ((remaining - 1) % childNbElements);
        
        auto && result = navigator.next(containerPtr, index, overjump);
        if((result == 0) && (overjump > 0) && not isAfterLast())
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
     * This is the case where the child hasn't a constant size. This means we
     * can not calculate the number of childjumps
     * 1. We move the child iterator as wide as possible
     * 2. we move this iterator one element
     * 3.
     */
    HDINLINE 
    uint
    gotoNext(uint const & jumpsize)
    {
        auto remaining = jumpsize;
        while(remaining > 0u and not isAfterLast())
        {
            remaining = childIterator.gotoNext(remaining);
            if(remaining == 0u)
                break;
            --remaining;
            navigator.next(containerPtr, index, 1u);
            if(not isAfterLast())
                childIterator.setToBegin(accessor.get(containerPtr, index));
            
        }
        return remaining;
    }


    HDINLINE 
    DeepIterator&
    operator-=(const uint & jumpsize)
    {
        auto tmpJump = jumpsize;
        if(isAfterLast())
        {
            --tmpJump;
            setToRbegin();
        }
        gotoPrevious(jumpsize);
        return *this;
    }
    
    template< 
        bool T = hasConstantSizeChild> 
    HDINLINE 
    uint
    gotoPrevious(uint const & jumpsize, typename std::enable_if<T == true>::type* = nullptr)
    {
        
        auto && remaining = childIterator.gotoPrevious(jumpsize);
        if(childIterator.isBeforeFirst() and remaining == 0)
        {
                remaining = jumpsize;
        }
        
        auto && childNbElements = childIterator.nbElements();
        auto && overjump = (remaining + childNbElements - 1) / childNbElements;
        int childJumps = ((remaining - 1) % childNbElements);
        
        auto && result = navigator.previous(containerPtr, index, overjump);
        if((result == 0u) && (overjump > 0u) && not isBeforeFirst())
        {
            childIterator.setToRbegin(accessor.get(containerPtr, index));
            childIterator -= childJumps;
        }
        // we only need to return something, if we are at the end
        uint const condition = (result > 0u);
        // the size of the jumps
        uint const notOverjumpedElements = (result-1u) * childNbElements;
        
        // The 1 is to set to the first element
        return condition * (notOverjumpedElements + childJumps + 1u);
    }
    
    template< 
        bool T = hasConstantSizeChild> 
    HDINLINE 
    uint 
    gotoPrevious(uint const & jumpsize, typename std::enable_if<T == false>::type* = nullptr)
    {
        auto remaining = jumpsize;
        while(remaining > 0u and not isBeforeFirst())
        {
            remaining = childIterator.gotoPrevious(remaining);
            if(remaining == 0u)
                break;
            --remaining;
            navigator.previous(containerPtr, index, 1u);
            if(not isBeforeFirst())
                childIterator.setToRbegin(accessor.get(containerPtr, index));
        }
        return remaining;
    }
    
    HDINLINE
    void
    setToBegin()
    {
        navigator.begin(containerPtr, index);
        if(not isBeforeFirst() and not isAfterLast())
        {
            childIterator.setToBegin((accessor.get(containerPtr, index)));
        }
    }

    HDINLINE
    void
    setToBegin(TContainer& con)
    {
        containerPtr = &con;
        setToBegin();

    }
    
    HDINLINE
    void
    setToBegin(TContainer* ptr)
    {
        containerPtr = ptr;
        setToBegin();

    }
    
    HDINLINE
    void
    setToEnd()
    {
        navigator.end(containerPtr, index);
        //childIterator.setToEnd((accessor.get(containerPtr, index)));
    }

    HDINLINE
    void
    setToEnd(TContainer* ptr)
    {
        containerPtr = ptr;
        setToEnd();
    }
    
    HDINLINE
    void
    setToRend()
    {
        navigator.rend(containerPtr, index);
        //childIterator.setToRend();
    }

    HDINLINE
    void
    setToRend(TContainer* ptr)
    {
        containerPtr = ptr;
        setToRend();
    }
    
    HDINLINE
    void 
    setToRbegin()
    {
        navigator.rbegin(containerPtr, index);
        if(not isBeforeFirst() && not isAfterLast())
        {
            childIterator.setToRbegin((accessor.get(containerPtr, index)));
        }
    }
    
    HDINLINE
    void 
    setToRbegin(ContainerRef con)
    {
        containerPtr = &con;
        setToRbegin();
    }
    
    HDINLINE
    void 
    setToRbegin(ContainerPtr ptr)
    {
        containerPtr = ptr;
        setToRbegin();
    }

    HDINLINE 
    bool
    isAfterLast()
    const
    {
        return navigator.isAfterLast(containerPtr, index);
    }
    
    HDINLINE 
    bool
    isBeforeFirst()
    const
    {
        return navigator.isBeforeFirst(containerPtr, index);
    }
    
    template<
        bool T = hasConstantSize>
    typename std::enable_if<T, int>::type
    nbElements()
    const
    {
        return childIterator.nbElements() * navigator.size(containerPtr);
    }
    
    
    /**
     * @brief This function calculate the distance from the current element to 
     * the end of the current Element
     * @return The distance to the next element in this layer
     */
    template<
        bool T = hasConstantSize>
    typename std::enable_if<T, int>::type
    getRangeToEnd()
    const
    {
        // The distance is the sum of two parameter: 
        // 1. the distance of the child (childDistance)
        // 2. the distance of the current layer * size of the childs (selfDistance)
        
        auto && childDistance = childIterator.getRangeToEnd();
        auto && selfDistance = (selfCompileTimeSize - index) * childIterator.nbElements();
        return childDistance + selfDistance;
    }
    
    
    ContainerPtr containerPtr = nullptr;
    IndexType index;
    ChildIterator childIterator;
    Navigator navigator;
    Accessor accessor;
private:

}; // struct DeepIterator





/** ************************************+
 * @brief The flat implementation 
 * ************************************/
template<
    typename TContainer, 
    typename TAccessor, 
    typename TNavigator>
struct DeepIterator<
    TContainer,     
    TAccessor, 
    TNavigator,
    hzdr::NoChild>
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
    
    typedef typename traits::ContainerCategory<ContainerType>::type     ContainerCategoryType;
    typedef typename traits::IndexType<ContainerType>::type   IndexType;
// container stuff

    
    static const bool isBidirectional = hzdr::traits::IsBidirectional<ContainerType>::value;
    static const bool isRandomAccessable = hzdr::traits::IsRandomAccessable<ContainerType>::value;
    static const bool hasConstantSize = traits::HasConstantSize<ContainerType>::value;
// functions 
public:


    HDINLINE DeepIterator() = default;
    HDINLINE DeepIterator(const DeepIterator&) = default;
    HDINLINE DeepIterator(DeepIterator&&) = default;

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
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor)),
        containerPtr(container)

        
    {

        setToBegin(container);
    }
    
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
        
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor)),
        containerPtr(container)
        
    {
        setToRbegin(container);
    }
    
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
        
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor)),
        containerPtr(container)
        
    {
        setToEnd(container);
    }
    
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
        
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor)),
        containerPtr(container)
        
    {
        setToRend(container);
    }
    
    /**
     * We need the template parameter for perfect forwarding
     */
    template<
        typename TAccessor_, 
        typename TNavigator_>
    HDINLINE
    DeepIterator(TAccessor_ && accessor, 
                 TNavigator_ && navigator):
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor))
    {}
    
    template<
        typename TAccessor_, 
        typename TNavigator_>
    HDINLINE
    DeepIterator(TAccessor_ && accessor, 
                 TNavigator_ && navigator,
                 hzdr::NoChild const &
                ):
        navigator(std::forward<TNavigator_>(navigator)),
        accessor(std::forward<TAccessor_>(accessor))
    {}
    
    
    
    /**
     * @brief goto the next element
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
            navigator.next(containerPtr, index,1);
        }
        return *this;
    }
    
    template<typename TJump, bool T=isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, DeepIterator&>::type
    operator+=(TJump const & jump)
    {
        auto tmpJump = jump;
        if(isBeforeFirst())
        {
            --tmpJump;
            setToBegin();
        }
        navigator.next(containerPtr, index, tmpJump);
        return *this;
    }
    
    template<typename TJump, bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, DeepIterator&>::type
    operator-=(TJump const & jump)
    {
        auto tmpJump = jump;
        if(isAfterLast())
        {
            --tmpJump;
            setToRbegin();
        }
        navigator.previous(containerPtr, index, jump);
        return *this;
    }
    
    template<typename TJump, bool T=isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, DeepIterator>::type
    operator+(TJump const & jump)
    {
        DeepIterator tmp = *this;
        tmp+=jump;
        return tmp;
    }
    
    template<typename TJump, bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, DeepIterator>::type
    operator-(TJump const & jump)
    {
        DeepIterator tmp = *this;
        tmp-=jump;
        return tmp;
    }
    
    template< bool T=isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, bool>::type
    operator<(DeepIterator const & other)
    {

        return accessor.lesser(
            containerPtr, 
            index,
            other.containerPtr,
            other.index);
    }
    
    template<bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, bool>::type
    operator>(DeepIterator const & other)
    {
        return accessor.greater(
            containerPtr,  
            index,
            other.containerPtr,
            other.index);
    }
    
    template< bool T=isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, bool>::type
    operator<=(DeepIterator const & other)
    {

        return *this < other || *this == other;
    }
    
    template<bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, bool>::type
    operator>=(DeepIterator const & other)
    {
        return *this > other || *this == other;
    }
    
    template<bool T = isRandomAccessable>
    HDINLINE
    typename std::enable_if<T, ComponentReference>::type
    operator[](IndexType const & index)
    {
        return accessor.get(containerPtr, index);
    }
    
    HDINLINE
    DeepIterator
    operator++(int)
    {
        DeepIterator tmp(*this);
        navigator.next(containerPtr, index,1);
        return tmp;
    }
    
    template<bool T = isBidirectional>
    HDINLINE 
    typename std::enable_if<T, DeepIterator& >::type
    operator--()
    {
        navigator.previous(containerPtr, index,1);
        return *this;
    }
    
    template<bool T = isBidirectional>
    HDINLINE 
    typename std::enable_if<T, DeepIterator >::type
    operator--(int)
    {
        DeepIterator tmp(*this);
        navigator.previous(containerPtr, index,1);
        return tmp;
    }
    
    HDINLINE
    auto 
    operator*()
    -> 
    ComponentReference
    {
        return (accessor.get(containerPtr, index));
    }
    
    
    
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
    
    HDINLINE 
    bool
    isAfterLast()
    const
    {
        return navigator.isAfterLast(containerPtr, index);
    }
    
    HDINLINE 
    bool
    isBeforeFirst()
    const
    {
        return navigator.isBeforeFirst(containerPtr, index);
    }
    
    HDINLINE
    bool
    operator==(const DeepIterator& other)
    const
    {
        return not (*this != other);
    }


    HDINLINE 
    void 
    setToBegin()
    {
        navigator.begin(containerPtr, index);
    }
    
    HDINLINE 
    void 
    setToBegin(ContainerPtr con)
    {
        containerPtr = con;
        navigator.begin(containerPtr, index);
    }
    
    HDINLINE 
    void 
    setToBegin(ContainerReference con)
    {
        containerPtr = &con;
        navigator.begin(containerPtr, index);
    }

    HDINLINE
    void 
    setToEnd(ContainerReference con)
    {
        containerPtr = &con;
        navigator.end(containerPtr, index);
    }
    
    HDINLINE
    void 
    setToEnd(ContainerPtr con)
    {
        containerPtr = con;
        navigator.end(containerPtr, index);
    }
    
    HDINLINE
    void 
    setToEnd()
    {
        navigator.end(containerPtr, index);
    }
    
    HDINLINE
    void 
    setToRend(ContainerReference con)
    {
        containerPtr = &con;
        navigator.rend(containerPtr, index);
    }
    
    HDINLINE
    void 
    setToRend(ContainerPtr con)
    {
        containerPtr = con;
        navigator.rend(containerPtr, index);
    }
    
    HDINLINE
    void 
    setToRend()
    {
        navigator.rend(containerPtr, index);
    }
    
    HDINLINE
    void 
    setToRbegin(ContainerReference con)
    {
        containerPtr = &con;
        navigator.rbegin(containerPtr, index);
    }
    
    HDINLINE
    void 
    setToRbegin(ContainerPtr con)
    {
        containerPtr = con;
        navigator.rbegin(containerPtr, index);
    }
    
    HDINLINE
    void 
    setToRbegin()
    {
        navigator.rbegin(containerPtr, index);
    }
    
    HDINLINE 
    auto 
    gotoNext(uint const & steps)
    ->
    uint
    {
        
        return navigator.next(containerPtr, index, steps);
    }
    
    HDINLINE
    auto 
    gotoPrevious(uint const & steps)
    ->
    uint
    {
        auto result = navigator.previous(containerPtr, index, steps);

        return result;
    }
    
    
//     template<
//         bool T = hasConstantSize>
//     typename std::enable_if<T, int>::type
//     getRangeToEnd()
//     const
//     {
//         return navigator.distanceToEnd(containerPtr, index);
//     }
    
//     template<
//         bool T = hasConstantSize>
//     typename std::enable_if<T, int>::type
//     getRangeToBegin()
//     const
//     {
//         return navigator.distanceToBegin(containerPtr, index);
//     }
    
    template<
        bool T = hasConstantSize>
    typename std::enable_if<T, int>::type
    nbElements()
    const
    {
        return navigator.size(containerPtr);
    }
    
    Navigator navigator;
    Accessor accessor;
    hzdr::NoChild childIterator;
    
protected:
    ContainerType* containerPtr = nullptr;
    IndexType index;

private:
}; // struct DeepIterator



namespace details 
{
template<
    typename TContainer,
    typename TChild,
// SFIANE Part
    typename TChildNoRef = typename std::decay<TChild>::type>
HDINLINE
auto
makeIterator( TChild && child, typename std::enable_if<std::is_same<TChildNoRef, hzdr::NoChild>::value>::type* = nullptr)
->
hzdr::NoChild
{
    return child;
}



/**
 * @brief bind an an iterator concept to an containertype. The concept has no child.
 * @tparam TContainer type of the container
 * @param concept an iterator concept
 * 
 */
template<
    typename TContainer,
    typename TConcept,
    typename TConceptNoRef = typename std::decay<TConcept>::type>
HDINLINE
auto 
makeIterator (
    TConcept && concept,
    typename std::enable_if<not std::is_same<TConceptNoRef, hzdr::NoChild>::value>::type* = nullptr)
->
DeepIterator<
        TContainer,
        decltype(makeAccessor<TContainer>(std::forward<TConcept>(concept).accessor)),
        decltype(makeNavigator<TContainer>(std::forward<TConcept>(concept).navigator)),
        decltype(makeIterator<
            typename traits::ComponentType<TContainer>::type>(std::forward<TConcept>(concept).child))>  
{
    typedef TContainer                                          ContainerType;

    typedef decltype(makeAccessor<ContainerType>(std::forward<TConcept>(concept).accessor))      AccessorType;
    typedef decltype(makeNavigator<ContainerType>(std::forward<TConcept>(concept).navigator))    NavigatorType;
    typedef decltype(makeIterator<
            typename traits::ComponentType<TContainer>::type>(std::forward<TConcept>(concept).child)) ChildType;


    typedef DeepIterator<
        ContainerType,
        AccessorType,
        NavigatorType,
        ChildType>         Iterator;
        
    return Iterator(
        makeAccessor<ContainerType>(std::forward<TConcept>(concept).accessor),
        makeNavigator<ContainerType>(std::forward<TConcept>(concept).navigator),
        makeIterator<typename traits::ComponentType<TContainer>::type>(std::forward<TConcept>(concept).child));
}

} // namespace details


/**
 * @brief Bind a container to a virtual iterator.  
 * @param con The container you like to inspect
 * @param iteratorConcept A virtual iterator, which describes the behavior of 
 * the iterator
 * @return An Iterator. It is set to the first element.
 */
template<
    typename TContainer,
    typename TAccessor,
    typename TNavigator,
    typename TChild,
    typename TIndex>
HDINLINE 
auto
makeIterator(
    TContainer && container,
    hzdr::details::IteratorConcept<
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
                typename std::decay<TContainer>::type>::type>(concept.childIterator))>         
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
        ChildType>         Iterator;
    
    return Iterator(
        container, 
        details::makeAccessor<ContainerType>(),
        details::makeNavigator<ContainerType>(concept.navigator),
        details::makeIterator<ComponentType>(concept.childIterator));
}



// template<
//     typename TContainer,
//     typename TNavigator,
//     typename TAccessor>
// HDINLINE
// auto 
// makeIterator(TContainer& container, 
//              TAccessor&& accessor, 
//              TNavigator&& navigator)
// -> hzdr::DeepIterator<
//     typename std::decay<TContainer>::type,  
//     TAccessor, 
//     TNavigator, 
//     hzdr::NoChild>
// {
//     typedef typename std::decay<TContainer>::type               ContainerType;
// 
// 
//     typedef hzdr::DeepIterator<ContainerType, TAccessor, TNavigator, hzdr::NoChild> Iterator;
//     return Iterator(container, 
//                     std::forward<TAccessor>(accessor), 
//                     std::forward<TNavigator>(navigator));
// }



#if 0

namespace details 
{
struct UndefinedType{};

template<
    typename TContainer,
    typename TNavigator,
    typename TAccessor
    >
HDINLINE
auto 
makeIterator(TAccessor&& accessor, 
              TNavigator&& navigator, 
              hzdr::NoChild&&)
-> hzdr::DeepIterator<TContainer,  
                      decltype(details::makeAccessor<TContainer>()), 
                      decltype(details::makeNavigator<TContainer>(std::forward<TNavigator>(navigator))), 
                      hzdr::NoChild>
{
    typedef decltype(details::makeNavigator<TContainer>(std::forward<TNavigator>(navigator))) IteratorNavigator;
    typedef decltype(details::makeAccessor<TContainer>()) IteratorAccessor;
    typedef hzdr::DeepIterator<TContainer,  
                            IteratorAccessor, 
                            IteratorNavigator, 
                            hzdr::NoChild> Iterator;
    return Iterator(details::makeAccessor<TContainer>(),
                    details::makeNavigator<TContainer>(std::forward<TNavigator>(navigator)));
}


template<
    typename TContainer,
    typename TNavigator,
    typename TAccessor,
    typename TChild
>
HDINLINE
auto 
makeIterator(TAccessor&& accessor, 
              TNavigator&& navigator, 
              TChild&& child)
-> hzdr::DeepIterator<TContainer,  
                      decltype(details::makeAccessor<TContainer>()), 
                      decltype(details::makeNavigator<TContainer>(std::forward<TNavigator>(navigator))), 
                      decltype(details::makeIterator< typename hzdr::traits::ComponentType<TContainer>::type>(std::move(child.accessor),
                                                   std::move(child.navigator),
                                                   std::move(child.childIterator)))>
{
    
    
    typedef decltype(details::makeNavigator<TContainer>(std::forward<TNavigator>(navigator))) IteratorNavigator;
    typedef decltype(details::makeAccessor<TContainer>()) IteratorAccessor;
    typedef typename hzdr::traits::ComponentType<TContainer>::type ChildContainerType;

    typedef decltype(details::makeIterator<ChildContainerType>(std::move(child.accessor),
                                                   std::move(child.navigator),
                                                   std::move(child.childIterator))) Child;
    

    
    auto && childIterator = details::makeIterator<ChildContainerType>(std::move(child.accessor),
                                                   std::move(child.navigator),
                                                   std::move(child.childIterator));
    
    
    typedef hzdr::DeepIterator<TContainer,  
                            IteratorAccessor, 
                            IteratorNavigator, 
                            Child> Iterator;    
        
    typedef decltype(details::makeAccessor<TContainer>()) AccessorType;
    typedef decltype(details::makeNavigator<TContainer>(std::move(navigator))) NavigatorType;
                    
    return Iterator(std::forward<AccessorType>(details::makeAccessor<TContainer>()),
                    std::forward<NavigatorType>(details::makeNavigator<TContainer>(std::move(navigator))),
                    childIterator);
}


} // namespace details


template<
    typename TContainer,
    typename TNavigator,
    typename TAccessor,
    typename TChildNavigator,
    typename TChildAccessor,
    typename TChildChild
    >
HDINLINE
auto 
makeIterator(TContainer& container, 
            TAccessor&& accessor, 
            TNavigator&& navigator, 
            hzdr::DeepIterator<details::UndefinedType, TChildAccessor, TChildNavigator, TChildChild>&& child)
-> hzdr::DeepIterator<typename std::remove_pointer<typename std::decay<TContainer>::type>::type,  
                      TAccessor, 
                      TNavigator, 
                      decltype(details::makeIterator<typename hzdr::traits::ComponentType<typename std::remove_pointer<typename std::decay<TContainer>::type>::type >::type>(std::move(child.accessor),
                                                   std::move(child.navigator),
                                                   std::move(child.childIterator)))
                     >

{
    typedef typename std::remove_pointer<typename std::decay<TContainer>::type>::type ContainerType;
    typedef typename hzdr::traits::ComponentType<ContainerType>::type ChildContainerType;

    typedef decltype(details::makeIterator<ChildContainerType>(std::move(child.accessor),
                                                   std::move(child.navigator),
                                                   std::move(child.childIterator))) Child;
    
    Child && childIterator = details::makeIterator<ChildContainerType>(std::move(child.accessor),
                                                   std::move(child.navigator),
                                                   std::move(child.childIterator));
    
    typedef hzdr::DeepIterator<ContainerType,  
                            TAccessor, 
                            TNavigator, 
                            Child> Iterator;    
                    
    return Iterator(container,
                    std::forward<TAccessor>(accessor),
                    std::forward<TNavigator>(navigator),
                    childIterator
                   );
    
}




template<
    typename TContainer,
    typename TNavigator,
    typename TAccessor
    >
HDINLINE
auto 
makeIterator(TContainer& container, 
             TAccessor&& accessor, 
             TNavigator&& navigator)
-> hzdr::DeepIterator<typename std::decay<TContainer>::type,  TAccessor, TNavigator, hzdr::NoChild>
{
    typedef typename std::decay<TContainer>::type ContainerType;
    typedef hzdr::DeepIterator<ContainerType,  TAccessor, TNavigator, hzdr::NoChild> Iterator;
    return Iterator(container, 
                    std::forward<TAccessor>(accessor), 
                    std::forward<TNavigator>(navigator));
}



#endif 

// template<typename TContainer,
//         typename TAccessor,
//         typename TNavigator,
//         typename TChild>
// HDINLINE
// auto
// makeChild(hzdr::DeepIterator<
//                 details::UndefinedType, 
//                 TAccessor, 
//                 TNavigator, 
//                 TChild>&& child
//           )
// -> hzdr::DeepIterator<
//     typename std::decay<TContainer>::type,
//         decltype(makeAccessor<
//             typename std::decay<TContainer>::type>(std::forward<TAccessor>(child.accessor))), 
//         decltype(makeNavigator<
//             typename std::decay<TContainer>::type>(std::forward<TNavigator>(child.navigator))),
//         decltype(makeChild<
//             typename traits::ComponentType<
//                 typename std::decay<TContainer>::type>::type >(std::forward<TChild>(child.childIterator)))>
// {
//     
//     typedef TContainer ContainerType;
//     typedef typename traits::ComponentType<ContainerType>::type ComponentType;
//     
//     typedef hzdr::DeepIterator<
//         ContainerType, 
//         decltype(makeAccessor<ContainerType>(
//             std::forward<TAccessor>(child.accessor))), 
//         decltype(makeNavigator<ContainerType>(
//             std::forward<TNavigator>(child.navigator))),
//         decltype(makeChild<ComponentType>(std::forward<TChild>(child.childIterator)))> Iterator;
//     
//     
//     return Iterator(
//         makeAccessor<ContainerType>(std::forward<TAccessor>(child.accessor)),
//         makeNavigator<ContainerType>(std::forward<TNavigator>(child.navigator)),
//         makeChild<ComponentType>(std::forward<TChild>(child.childIterator)));
// }




/*
namespace details 
{
template<typename Iterator>
struct ChildTraits;
    
template<
    template<typename, typename, typename, typename> typename TIterator,
    typename TContainer,
    typename TAccessor, 
    typename TNavigator,
    typename TChild>
struct ChildTraits<TIterator<TContainer, TAccessor, TNavigator, TChild> >
{
        typedef TContainer  ContainerType;
        typedef TAccessor  AccessorType;
        typedef TNavigator  NavigatorType;
        typedef TChild      ChildType;
};

template<
    template<typename, typename, typename> typename TIterator,
    typename TAccessor, 
    typename TNavigator,
    typename TChild>
struct ChildTraits<TIterator<TAccessor, TNavigator, TChild> >
{
        typedef TAccessor   AccessorType;
        typedef TNavigator  NavigatorType;
        typedef TChild      ChildType;
};

} // namespace details

template<
    typename TContainer,
    typename TIterator,
    typename TIteratorFree = typename std::decay<TIterator>::type,
    typename TAccessor = typename details::ChildTraits<TIterator>::AccessorType, 
    typename TNavigator = typename details::ChildTraits<TIterator>::NavigatorType,
    typename TChild = typename details::ChildTraits<TIterator>::ChildType,
// SFIANE part
    typename T = typename std::enable_if<std::is_same<TChild, hzdr::NoChild>::value>::type
    >
HDINLINE
auto
makeChild(TIterator && child)
-> 
hzdr::DeepIterator<
    TContainer,
    decltype(
        makeAccessor<TContainer>(child.accessor)),
    decltype(
        makeNavigator<TContainer>(child.navigator)),
    hzdr::NoChild>
{

    typedef TContainer ContainerType;
    typedef hzdr::DeepIterator<
        ContainerType, 
        decltype(
            makeAccessor<ContainerType>(child.accessor)),
        decltype(
            makeNavigator<ContainerType>(child.navigator)),
        hzdr::NoChild> Iterator;
    
    auto && accessor = makeAccessor<TContainer>(child.accessor);
    auto && navigator = makeNavigator<TContainer>( child.navigator);
    
    return Iterator(
        accessor,
        navigator);

}

template<
    typename TContainer,
    typename TIterator,
    typename ComponentType = typename hzdr::traits::ComponentType<TContainer>::type,
    typename TIteratorFree = typename std::decay<TIterator>::type,
    typename TAccessor = typename details::ChildTraits<TIteratorFree>::AccessorType, 
    typename TNavigator = typename details::ChildTraits<TIteratorFree>::NavigatorType,
    typename TChild = typename details::ChildTraits<TIteratorFree>::ChildType,
// SFIANE part
    typename T = typename std::enable_if<not std::is_same<TChild, hzdr::NoChild>::value>::type
    >
HDINLINE
auto
makeChild(TIterator && child)
-> 
hzdr::DeepIterator<
    TContainer,
    decltype(
        makeAccessor<TContainer>(
            std::forward<TAccessor>(child.accessor))),
    decltype(
        makeNavigator<TContainer>( 
            std::forward<TNavigator>(child.navigator))),
    decltype(makeChild<ComponentType>(std::forward<TChild>(child.child)))>
{

    typedef TContainer ContainerType;
    typedef hzdr::DeepIterator<
        ContainerType, 
        decltype(
            makeAccessor<ContainerType>(
                std::forward<TAccessor>(child.accessor))),
        decltype(
            makeNavigator<ContainerType>( 
                std::forward<TNavigator>(child.navigator))),
        decltype(makeChild<ComponentType>(std::forward<TChild>(child.child)))> Iterator;
    
        
    auto && accessor = makeAccessor<TContainer>(
                std::forward<TAccessor>(child.accessor));
    auto && navigator = makeNavigator<TContainer>( 
                std::forward<TNavigator>(child.navigator));
    auto && childNew = makeChild<ComponentType>(std::forward<TChild>(child.child));
    
    return Iterator(
        accessor,
        navigator,
        childNew);

}


*/


}// namespace hzdr
/*
template< 
    typename TContainer,
    typename TAccessor,
    typename TNavigator,
    typename TChild>
std::ostream& operator<<(std::ostream& out, const hzdr::DeepIterator<TContainer, TAccessor, TNavigator, TChild>& iter)
{
    out << "[" << iter.containerPtr << ", " << iter.componentPtr << ", " << iter.index << "," << iter.childIterator << "]";
    return out;
    
}

template< 
    typename TContainer,
    typename TAccessor,
    typename TNavigator>
std::ostream& operator<<(std::ostream& out, const hzdr::DeepIterator<TContainer, TAccessor, TNavigator, hzdr::NoChild>& iter)
{
    out << "[" << iter.containerPtr << ", " << iter.componentPtr << ", " << iter.index << "]";
    return out;
    
}*/


