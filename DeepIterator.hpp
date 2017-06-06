/**
 * @author Sebastian Hahn (t.hahn@hzdr.de )
 * @brief The DeepIterator class is used to iterator over interleaved data 
 * structures. The simplest example is for an interleaved data structure is 
 * std::vector< std::vector< int > >. The deepiterator iterates over all ints 
 * within the structure.
 * The iterator support lists and index based access. Both are specialised. 
 * Because the implementation of interleaved and flat is different, we need four
 * implemtations of the iterator:
 * 1. flat and list based
 * 2. flat and index based
 * 3. interleaved and list based
 * 4. interleaved and index based
 *
 * The iterator use the trait \b IsIndexable to decide wether the datastructure 
 * is array like or list like. 
 * This implementation is special for the datastructurs of PIConGPU. 

 */
#include <sstream>
#pragma once
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "PIC/Supercell.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Collective.hpp"
#include "Traits/NeedRuntimeSize.hpp"
#include <boost/iterator/iterator_concepts.hpp>
#include "Iterator/Wrapper.hpp"
#include "Traits/IsIndexable.hpp"
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
         typename TCollective, 
         typename TRuntimeVariables,
         typename TChild,
         typename TEnable = void>
struct DeepIterator;





/** ************************************+
 * @brief The flat implementation 
 * ************************************/
template<typename TContainer,
        typename TAccessor, 
        typename TNavigator,
        typename TRuntimeVariables,
        typename TCollective>
struct DeepIterator<TContainer, 
                    TAccessor, 
                    TNavigator, 
                    TCollective, 
                    TRuntimeVariables,
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
    typedef TCollective                                                 Collecter;
    typedef Wrapper< ComponentType, TCollective>                        WrapperType;
// functions 
//    static_assert(std::is_same<typename TAccessor::ReturnType, ComponentType>::value, "Returntype of accessor must be the same as Valuetype of TContainer");
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */
    HDINLINE
    DeepIterator(nullptr_t)
    {}

    HDINLINE
    DeepIterator(ContainerType* _ptr, 
                 const int_fast32_t& nbElemsInLast, 
                 const TRuntimeVariables& runtimeVariables
                ):
        runtimeVariables(runtimeVariables)
    {
        if(coll.isMover())
        {
            Navigator::first(_ptr, containerPtr, componentPtr, index, runtimeVariables);
        }
        coll.sync();
    }
    
    /**
     * @brief goto the next element
     */

    HDINLINE
    DeepIterator&
    operator++()
    {

        coll.sync();
        if(coll.isMover())
        {
            Navigator::next(containerPtr, componentPtr, index, runtimeVariables);
        }
        coll.sync();

        return *this;
    }
    
    HDINLINE
    WrapperType
    operator*()
    {

        const auto nbElem = traits::NeedRuntimeSize<ContainerType>::test(containerPtr) * runtimeVariables.getNbElements() 
                          + (1 - traits::NeedRuntimeSize<ContainerType>::test(containerPtr)) * traits::NumberElements<ContainerType>::value;
        return WrapperType(Accessor::get(containerPtr, componentPtr, index, nbElem));
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
    operator!=(nullptr_t)
    const
    {
        return componentPtr != nullptr or containerPtr != nullptr;
    }
    
    
    HDINLINE    
    bool
    operator!=(const int_fast32_t& maxIndex )
    const
    {
        return index < maxIndex 
            and index >= 0;
    }
    
protected:
    Collecter coll;
    ComponentType* componentPtr = nullptr; 
    ContainerType* containerPtr = nullptr;
    int_fast32_t index = std::numeric_limits<int_fast32_t>::min();
    TRuntimeVariables runtimeVariables;
private:
}; // struct DeepIterator







/** ************************************+
 * @brief The nested implementation
 * ************************************/
template<typename TContainer,
        typename TAccessor, 
        typename TNavigator,
        typename TCollective,
        typename TRuntimeVariables,
        typename TChild>
struct DeepIterator<TContainer, 
                    TAccessor, 
                    TNavigator, 
                    TCollective, 
                    TRuntimeVariables,
                    TChild,
                    typename std::enable_if<not std::is_same<TChild, hzdr::NoChild>::value>::type >
{
// datatypes
    
public:
    typedef TContainer                                                  ContainerType;
    typedef typename hzdr::traits::ComponentType<ContainerType>::type  ComponentType;

    typedef ComponentType*                              ComponentPointer;
    typedef ComponentType&                              ComponentReference;
    typedef TAccessor                               Accessor;
    typedef TNavigator                              Navigator;
    typedef TCollective                             Collecter;

    typedef traits::NeedRuntimeSize<ContainerType>    RuntimeSize;
// child things
    typedef TChild                                  ChildView;
    typedef typename TChild::Iterator               ChildIterator;
    typedef typename ChildIterator::ReturnType      ReturnType;
    typedef typename ChildIterator::WrapperType     WrapperType;

    // tests
  //  static_assert(std::is_same<typename TAccessor::ReturnType, ComponentType>::value, "Returntype of accessor must be the same as Valuetype of TContainer");
    
    // functions
    
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */
    HDINLINE
    DeepIterator(nullptr_t, 
                 const int_fast32_t& nbElems
    )
    {}
    
    HDINLINE
    DeepIterator()
        {}
    HDINLINE
    DeepIterator(ContainerType* _ptr)
    {
        if(coll.isMover())
        {
                    
            Navigator::first(_ptr, containerPtr, index, runtimeVariables);

        }
        coll.sync();
        
    }
    
    HDINLINE
    DeepIterator(ContainerType* ptr2, 
                 TRuntimeVariables runtimeVariables,
                 ChildView view):
                 
                 childView(view),
                 childIter(view.begin()),
                 runtimeVariables(runtimeVariables)
                
                 
    {

//        if(coll.isMover())
//        {
            Navigator::first(ptr2, containerPtr, componentPtr, index, runtimeVariables);

 //       }
        const auto nbElem = traits::NeedRuntimeSize<ContainerType>::test(containerPtr) * runtimeVariables.getNbElements() 
                          + (1 - traits::NeedRuntimeSize<ContainerType>::test(containerPtr)) * traits::NumberElements<ContainerType>::value;
        childView.setPtr(ChildView(Accessor::get(containerPtr, componentPtr, index, nbElem)));
        childIter = childView.begin();
    }
    
    /**
     * @brief goto the next element
     */
    HDINLINE
    DeepIterator&
    operator++()
    {
        coll.sync();
   //     if(coll.isMover())
   //     {

             ++childIter;
            if(not (childIter != childView.end()))
            {
                Navigator::next(containerPtr, componentPtr, index, runtimeVariables);
                const auto nbElem = traits::NeedRuntimeSize<ContainerType>::test(containerPtr) * runtimeVariables.getNbElements() 
                          + (1 - traits::NeedRuntimeSize<ContainerType>::test(containerPtr)) * traits::NumberElements<ContainerType>::value;
                childView.setPtr(ChildView(Accessor::get(containerPtr, componentPtr, index, nbElem)));
                childIter = childView.begin();
            }
                
    //    }
        coll.sync();
        return *this;
    }
    
    HDINLINE
    WrapperType
    operator*()
    {
      //  auto t = Accessor::get(componentPtr, index);
//        std::cout << "ChildIter" << std::endl;
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
        return (componentPtr != nullptr) or (containerPtr != nullptr);
    }
    

    HDINLINE    
    bool
    operator!=(const int_fast32_t& maxIndex )
    const
    {
        return index < maxIndex 
            and index >= 0;
    }
    
protected:
    Collecter coll;
    TContainer* containerPtr = nullptr;
    int_fast32_t index;
    ComponentType* componentPtr= nullptr;
    ChildView childView;
    ChildIterator childIter;
    TRuntimeVariables runtimeVariables;
    
private:

}; // struct DeepIterator

}// namespace hzdr
