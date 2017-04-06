/**
 * @author Sebastian Hahn (t.hahn@hzdr.de )
 * @brief The View provides functionality for the DeepIterator. The first 
 * one is the construction of the DeepIterator type. This includes the navigator
 * and the accessor. The second part of the functionality is providing the begin
 * and end functions.
 * The import template arguments are TContainer and TElement. 
 * 
 */

#pragma once
#include "DeepIterator.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Collective.hpp"
#include "Traits/IsIndexable.hpp"
#include "Traits/NumberElements.hpp"
#include "Traits/HasJumpSize.hpp"
#include "Traits/HasNbRuntimeElements.hpp"
#include "Traits/HasOffset.hpp"
#include <type_traits>
namespace hzdr 
{
    
   
/** *********************************************
 * @brief This view is the connector between two layers. 
 * @tparam TContainer The datatype of the input type: At the moment we have
 * Frame and Supercell Implemented.
 * @tparam TElement The return type. Implemented are particle frames and container
 * @tparam TDirection The direction of the iteration. 
 * @tparam TRuntimeVariables A tupble with variables which are know at runtime. 
 * This should be a struct with the following possible public accessable vari-
 * ables: 
 * 1. nbRuntimeElements: If the datastructure has a size, which is firstly known
 * at runtime, you can use this variable to specify the size
 * 2. offset: First position after begin. In other word: these number of elements 
 * are ignored. This is needed if you write parallel code. If this value is not
 * given, offset=1 will be assumed.
 * 3. jumpsize: Size of the jump when called ++, i.e. What is the next element. 
 * If this value is not given, jumpsize=1 will be assumed.
 * ********************************************/
template<
    typename TElement,
    hzdr::Direction TDirection,
    typename TColl,
    typename TRuntimeVariables,
    typename TChild = NoChild>
struct View
{
public:
// Datatypes    
    typedef TElement                                                                                            ValueType; 
    typedef ValueType*                                                                                          ValuePointer;
    typedef TChild                                                                                              ChildType; 
    typedef typename std::conditional<traits::IsIndexable<ValueType>::value, Indexable, ValueType>::type        IndexableType;
    typedef Navigator<IndexableType, TDirection, 0>                                                             NavigatorType;
    typedef Accessor<IndexableType>                                                                             AccessorType;
    typedef DeepIterator<ValueType, AccessorType, NavigatorType, TColl, TRuntimeVariables, ChildType>           iterator; 
    typedef iterator                                                                                            Iterator; 
    typedef traits::NeedRuntimeSize<TElement>                                                                   RuntimeSize;
    typedef TRuntimeVariables                                                                                   RunTimeVariables;
    
public:
    
    
/***************
 * constructors without childs
 **************/
    View():
        ptr(nullptr)
        {}
        /**
     * @param container The element 
     */
    View(ValueType& container):
        ptr(&container)
    {}
    
    View(ValuePointer con):
        ptr(con)
    {}
    
    View(const View& other) = default;
    View(View&& other) = default;
        
   View(ValueType& container, const RunTimeVariables& runtimeVar):
        runtimeVars(runtimeVar),
        ptr(&container)
    {
        
    }
    
    View(ValuePointer con, const RunTimeVariables& runtimeVar):
        runtimeVars(runtimeVar),
        ptr(con)
    {}
    
    
    
    
    
    /**
     * @param container The element 
     */
    View(ValueType& container, ChildType& child):
        ptr(&container), 
        childView(child)
    {}
    
    View(ValuePointer con,ChildType& child):
        ptr(con),
        childView(child)
        {}
        
        
   View(ValueType& container, const RunTimeVariables& runtimeVar, const ChildType& child):
        runtimeVars(runtimeVar),
        ptr(&container), 
        childView(child)
    {
        
    }
    
    View(ValuePointer con, const RunTimeVariables& runtimeVar, ChildType&& child):
        runtimeVars(runtimeVar),
        ptr(con),
        childView(child)
    {}
    
    
        
    View& operator=(View&& other)
    {
        std::swap(ptr, other.ptr);
        return *this;
    }
    
    View& operator=(const View& other)
    {
        ptr = other.ptr;
        
        return *this;
    }
    
    /**
     * 1. Iterator with runtime and offset
     */
    
    template< bool test =  std::is_same<ChildType, NoChild>::value>
    typename std::enable_if<test, Iterator>::type 
    begin() 
    {
       const auto t =traits::NumberElements< TElement>::value;
       return Iterator(ptr, t, runtimeVars);
    }
    
    
    template< bool test = not std::is_same<ChildType, NoChild>::value, typename TUnused =void>
    typename std::enable_if<test, Iterator>::type                                       
    begin() 
    {
       const auto t =traits::NumberElements< TElement>::value;
       return Iterator(ptr, t, runtimeVars, childView);
    }
    
    /*
    Iterator begin() {
        
        if(hzdr::traits::HasNbRuntimeElementst<RunTimeVariables>::value)
        {
            if(hzdr::traits::HasOffset<RunTimeVariables>::value)
            {
                return Iterator(ptr, runtimeVars.offset, t, runtimeVars.nbRuntimeElements, childView);
            }
            else 
            {
                ;
            }
        }
        else 
        {
            if(hzdr::traits::HasOffset<RunTimeVariables>::value)
            {
                return Iterator(ptr, runtimeVars.offset, t, 0, childView);
            }
            else 
            {
                return Iterator(ptr, 0, t, 0, childView);
            }
        }
    }
*/
    Iterator end() {
            const uint_fast32_t elem = RuntimeSize::test(ptr)? runtimeVars.getNbElements()  : traits::NumberElements< TElement>::value;
            return Iterator(nullptr, elem);
    }

    
    
protected:

    RunTimeVariables runtimeVars;
    ValuePointer ptr;
    ChildType childView;
}; // struct View

#if 0
/** ****************************************************************************
 *@brief specialisation for Particle in frames
 ******************************************************************************/

template<
    typename TPos,
    hzdr::Direction TDirection,
    size_t jumpSize,
    uint32_t Dim,
    typename TCollective,
    uint32_t nbParticleInFrame
    >
struct View<
            hzdr::Frame<Particle<TPos, Dim>, nbParticleInFrame>, 
            hzdr::Particle<TPos, Dim>, 
            TDirection, 
            TCollective,
            jumpSize>
{
    typedef Particle<TPos, Dim>                                                                             ValueType;
    typedef ValueType                                                                                       ReturnType;
    typedef Frame<ValueType, nbParticleInFrame>                                                             FrameType;
    typedef FrameType                                                                                       InputType;
    typedef typename std::conditional<traits::IsIndexable<FrameType>::value, Indexable, FrameType>::type    IndexableType;
    typedef Navigator<IndexableType, TDirection, jumpSize>                                                  NavigatorType;
    typedef Accessor<IndexableType>                                                                         AccessorType;
    typedef DeepIterator<FrameType, AccessorType, NavigatorType, TCollective,hzdr::NoChild>                 iterator;
    typedef DeepIterator<FrameType, AccessorType, NavigatorType, TCollective,hzdr::NoChild>                 Iterator;
    /**
     * FrameType 
     */
    
    View(FrameType* container, uint32_t nbElem):
        conPtr(container), nbElem(nbElem)
    {}
    
    View(FrameType& container, uint32_t nbElem):
        conPtr(&container), nbElem(nbElem)
    {}
    
    View(const View& other):
        conPtr(other.conPtr),
        nbElem(other.nbElem)
    {}
    
    View(nullptr_t, uint32_t):
        conPtr(nullptr), 
        nbElem(0)
        {}
    
    View& operator=(const View&) = default;
    
    iterator begin() {
        return iterator(*conPtr, 0);
    }
    
    template<typename TOffset>
    iterator begin(const TOffset& offset) {
        return iterator(*conPtr, offset);
    }
    
    
    iterator end() {
        if(conPtr->nextFrame != nullptr)
        {
            return iterator(nbParticleInFrame);
        }
        else
        {
            return iterator(nbElem);
        }
    }
    
    FrameType* conPtr;
    uint32_t nbElem;
}; // 

/** ****************************************************************************
 *@brief specialisation for Frames in Suprecell
 ******************************************************************************/

template<
    typename TParticle,
    hzdr::Direction TDirection,
    size_t jumpSize,
    uint32_t nbParticleInFrame,
    typename TCollective
    >
struct View<
        hzdr::SuperCell<hzdr::Frame<TParticle, nbParticleInFrame> >,
        hzdr::Frame<TParticle, nbParticleInFrame>,
        TDirection,
        TCollective,
        jumpSize
        >
{
    typedef Frame<TParticle, nbParticleInFrame>                                                         ValueType;
    typedef SuperCell<ValueType >                                                                       ContainerType;
    typedef Navigator<ContainerType, TDirection, jumpSize>                                              NavigatorType;
    typedef Accessor<ContainerType>                                                                     AccessorType;
    typedef DeepIterator<ContainerType, AccessorType, NavigatorType, TCollective,  hzdr::NoChild>       iterator;
    typedef DeepIterator<ContainerType, AccessorType, NavigatorType, TCollective,  hzdr::NoChild>       Iterator;
    typedef View<ContainerType, ValueType, TDirection, hzdr::Collectivity::NonCollectiv, jumpSize>  ThisType;
    typedef ContainerType                                                                               InputType;
    
    View():
        conPtr(nullptr),
        nbElem(0)
    {}
    
    View(ContainerType& container):
        conPtr(&container)
    {}
    
    View(ContainerType* container):
        conPtr(container)
    {}
    
    View& operator=(const View&) = default;
    
    iterator begin() {
        return iterator(conPtr->firstFrame);
    }
    
    template<typename TOffset>
    iterator begin(const TOffset& offset) {
        return iterator(conPtr->firstFrame, offset);
    }
    
    iterator end() {
        return iterator(nullptr);
    }
    
    ContainerType* conPtr;
    uint32_t nbElem;
}; // 
#endif
} // namespace hzdr