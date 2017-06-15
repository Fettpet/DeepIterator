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
#include "Traits/NumberElements.hpp"
#include "Traits/HasJumpSize.hpp"
#include "Traits/HasNbRuntimeElements.hpp"
#include "Traits/HasOffset.hpp"
#include "Definitions/hdinline.hpp"
#include <type_traits>
namespace hzdr 
{
    
   
/** *********************************************
 * @brief This view is the connector between two layers. 
 * @tparam TElement The input container. It must have a typedef ValueType which 
 * is the typedef of elements within the container. The next requirement is that
 * the traits NeedRuntimeSize and NumberElements are specified for this container.
 * @tparam TDirection The direction of the iteration. There are to posibilities:
 * 1. Forward and; 2. Backward \see Policies.hpp 
 * @tparam TCollective is used to determine the collective properties of your 
 * iterator.
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
 * @tparam 
 * ********************************************/
template<
    typename TContainer,
    hzdr::Direction TDirection,
    typename TCollective,
    typename TRuntimeVariables,
    typename TChild = NoChild>
struct View
{
public:
// Datatypes    
    typedef TContainer                                                                                              ContainerType;
    typedef ContainerType*                                                                                          ContainerPtr;
    typedef typename traits::ComponentType<ContainerType>::type                                                     ComponentType;
    typedef TChild                                                                                                  ChildType; 
    typedef Navigator<ContainerType, TDirection>                                                                    NavigatorType;
    typedef Accessor<ContainerType>                                                                                 AccessorType;
    typedef Wrapper< ComponentType, TCollective>                                                                    WrapperType;
    typedef DeepIterator<ContainerType, AccessorType, NavigatorType, WrapperType,
                                                                     TCollective, TRuntimeVariables, ChildType>     iterator; 
    typedef iterator                                                                                                Iterator; 
    typedef traits::NeedRuntimeSize<TContainer>                                                                     RuntimeSize;
    typedef TRuntimeVariables                                                                                       RunTimeVariables;
    
public:
    
    
/***************
 * constructors without childs
 **************/
    HDINLINE
    View():
        ptr(nullptr)
        {}
        /**
     * @param container The element 
     */
        
    HDINLINE
    View(ContainerType& container):
        ptr(&container)
    {}
    HDINLINE
    View(ContainerPtr con):
        ptr(con)
    {}
    HDINLINE
    View(const View& other) = default;
    HDINLINE
    View(View&& other) = default;
     
    HDINLINE
    View(ContainerType& container, const RunTimeVariables& runtimeVar):
        runtimeVars(runtimeVar),
        ptr(&container)
    {
        
    }
    
    HDINLINE
    View(ContainerPtr con, const RunTimeVariables& runtimeVar):
        runtimeVars(runtimeVar),
        ptr(con)
    {}
    
    template<typename AccessorPointer>
    HDINLINE
    View(View oldView, AccessorPointer* accPtr):
        runtimeVars(oldView.runtimeVars),
        ptr(accPtr),
        childView(oldView.childView)
        {}
    
    
    /**
     * @param container The element 
     */
    HDINLINE
    View(ContainerType& container, ChildType& child):
        ptr(&container), 
        childView(child)
    {}
    
    HDINLINE
    View(ContainerPtr con,ChildType& child):
        ptr(con),
        childView(child)
        {}
        
    HDINLINE  
    View(ContainerType& container, const RunTimeVariables& runtimeVar, const ChildType& child):
        runtimeVars(runtimeVar),
        ptr(&container), 
        childView(child)
    {
        
    }
    
    HDINLINE
    View(ContainerPtr con, const RunTimeVariables& runtimeVar, ChildType&& child):
        runtimeVars(runtimeVar),
        ptr(con),
        childView(child)
    {}
    
    
    HDINLINE    
    View& operator=(View&& other) = default;

    
    HDINLINE
    View& operator=(const View& other) = default;
    
    /**
     * 1. Iterator with runtime and offset
     */
    
    template< bool test =  std::is_same<ChildType, NoChild>::value>
    HDINLINE
    typename std::enable_if<test, Iterator>::type 
    begin() 
    {
       const auto t =traits::NumberElements< ContainerType>::value;
       return Iterator(ptr, t, runtimeVars);
    }
    
    
    template< bool test = not std::is_same<ChildType, NoChild>::value, typename TUnused =void>
    HDINLINE
    typename std::enable_if<test, Iterator>::type                                       
    begin() 
    {
       return Iterator(ptr, runtimeVars, childView);
    }
    

    
    
    HDINLINE
    nullptr_t
    end() {
            return nullptr;
    }

    HDINLINE
    void
    setPtr(ContainerPtr _ptr)
    {
        ptr = _ptr;
    }
    
    HDINLINE
    void
    setPtr(const View& other)
    {
        ptr = other.ptr;
    }
    
 protected:

    RunTimeVariables runtimeVars;
    ContainerPtr ptr;
    ChildType childView;
}; // struct View


} // namespace hzdr
