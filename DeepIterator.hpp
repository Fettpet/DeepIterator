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
 Wrapper(accessor.get(containerPtr, componentPtr, index, runtimeVariables),
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
#include "Iterator/Policies.hpp"
#include "Iterator/Collective.hpp"
#include <boost/iterator/iterator_concepts.hpp>
#include "Iterator/Wrapper.hpp"
#include <limits>
#include <cassert>
#include <type_traits>
#include "Traits/NumberElements.hpp"
#include "Traits/Componenttype.hpp"
#include "Definitions/hdinline.hpp"


namespace hzdr 
{
/**
 * @tparam TContainer is the type of the container 
 * @tparam TChild ist ein virtueller Container oder NoChild
 */

template<typename TContainer, 
         typename TAccessor, 
         typename TNavigator, 
         typename TWrapper,
         typename TChild,
         typename TEnable = void>
struct DeepIterator;





/** ************************************+
 * @brief The flat implementation 
 * ************************************/
template<typename TContainer,
        typename TAccessor, 
        typename TNavigator,
        typename TWrapper>
struct DeepIterator<TContainer, 
                    TAccessor, 
                    TNavigator,
                    TWrapper,
                    hzdr::NoChild,
                    void>
{
// datatypes
public:
    typedef TContainer                                                  ContainerType;
    typedef typename hzdr::traits::ComponentType<ContainerType>::type   ComponentType;
    typedef ComponentType*                                              ComponentPointer;
    typedef ComponentType&                                              ComponentReference;
    typedef ComponentType                                               ReturnType;
    typedef TAccessor                                                   Accessor;
    typedef TNavigator                                                  Navigator;
    typedef TWrapper                                                    WrapperType;
// functions 
//    static_assert(std::is_same<typename Taccessor.ReturnType, ComponentType>::value, "Returntype of accessor must be the same as Valuetype of TContainer");
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */
    HDINLINE
    DeepIterator(nullptr_t)
    {}
    
    HDINLINE
    DeepIterator()
    {}

    HDINLINE
    DeepIterator(ContainerType* _ptr, const uint_fast32_t& offset)
    {

        navigator.first(_ptr, containerPtr, componentPtr, index,  offset);
    }
    
    HDINLINE DeepIterator(const DeepIterator&) = default;
    
    /**
     * @brief goto the next element
     */

    HDINLINE
    DeepIterator&
    operator++()
    {


        navigator.next(containerPtr, componentPtr, index);


        return *this;
    }
    
    HDINLINE
    WrapperType
    operator*()
    {


        return WrapperType(accessor.get(containerPtr, componentPtr, index),
                        containerPtr, componentPtr, index);
    }
    
    template<typename TIndex>
    HDINLINE 
    DeepIterator
    operator+(const TIndex& jumpsize)
    const 
    {
        DeepIterator result(*this);
        for(TIndex i=static_cast<TIndex>(0); i< jumpsize; ++i)
            ++result;
        return result;
    }
    
    HDINLINE
    bool
    operator!=(const DeepIterator& other)
    const
    {
        return componentPtr != other.componentPtr
            or containerPtr != other.containerPtr
            or index != other.index;
    }
    
    HDINLINE
    bool
    operator==(const DeepIterator& other)
    const
    {
        return not (*this != other);
    }

    HDINLINE    
    bool
    operator!=(nullptr_t)
    const
    {
         return not navigator.isEnd(containerPtr, componentPtr, index);
    }
    
    HDINLINE    
    bool
    operator==(nullptr_t)
    const
    {
         return navigator.isEnd(containerPtr, componentPtr, index);
    }
    
// protected:
    ComponentType* componentPtr = nullptr; 
    ContainerType* containerPtr = nullptr;
    int_fast32_t index = std::numeric_limits<int_fast32_t>::min();
    Navigator navigator;
    Accessor accessor;
private:
}; // struct DeepIterator







/** ************************************+
 * @brief The nested implementation
 * ************************************/
template<typename TContainer,
        typename TAccessor, 
        typename TNavigator,
        typename TChild,
        typename TWrapper>
struct DeepIterator<TContainer, 
                    TAccessor, 
                    TNavigator, 
                    TWrapper,
                    TChild,
                    typename std::enable_if<not std::is_same<TChild, hzdr::NoChild>::value>::type >
{
// datatypes
    
public:
    typedef TContainer                                                  ContainerType;
    typedef typename hzdr::traits::ComponentType<ContainerType>::type   ComponentType;

    typedef ComponentType*                                              ComponentPointer;
    typedef ComponentType&                                              ComponentReference;
    typedef TAccessor                                                   Accessor;
    typedef TNavigator                                                  Navigator;

// child things
    typedef TChild                                                      ChildView;
    typedef typename ChildView::Iterator                                ChildIterator;
    typedef typename ChildView::WrapperType                             WrapperType;

    
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */

    
    template<typename TIndex>
    HDINLINE 
    DeepIterator
    operator+(const TIndex& jumpsize)
    const 
    {
        DeepIterator result(*this);
        for(TIndex i=static_cast<TIndex>(0); i< jumpsize; ++i)
            ++result;
        return result;
    }
    
    HDINLINE
    DeepIterator(ContainerType* ptr2, const uint_fast32_t& offset)
    {

        navigator.first(ptr2, containerPtr, componentPtr, index, offset);
        childView = ChildView(accessor.get(containerPtr, componentPtr, index));

        childIter = childView.begin();
    }
    
        HDINLINE
    DeepIterator(ContainerType* ptr2, const ChildView& child,const uint_fast32_t& offset)
    {

        navigator.first(ptr2, containerPtr, componentPtr, index, offset);
        childView = ChildView(child, accessor.get(containerPtr, componentPtr, index));

        childIter = childView.begin();
    }
    
    /**
     * @brief goto the next element
     */
    HDINLINE
    DeepIterator&
    operator++()
    {


        ++childIter;
        if(not (childIter != childView.end()))
        {
            navigator.next(containerPtr, componentPtr, index);
            childView = ChildView(childView, accessor.get(containerPtr, componentPtr, index));
            childIter = childView.begin();
        }

        return *this;
    }
    
    HDINLINE
    WrapperType
    operator*()
    {

        return *childIter;
    }
    
    HDINLINE
    bool
    operator!=(const DeepIterator& other)
    const
    {
        
        return componentPtr != other.componentPtr
            or containerPtr != other.containerPtr
            or index != other.index 
            or other.childIter != childIter;
    }

    HDINLINE    
    bool
    operator!=(nullptr_t)
    const
    {
        
        return not navigator.isEnd(containerPtr, componentPtr, index);
    }
    


// protected:
    TContainer* containerPtr = nullptr;
    int_fast32_t index;
    ComponentType* componentPtr= nullptr;
    ChildView childView;
    ChildIterator childIter;
    Navigator navigator;
    Accessor accessor;
private:

}; // struct DeepIterator

}// namespace hzdr
