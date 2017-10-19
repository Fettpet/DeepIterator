#pragma once

/**
 * \struct View
 * @author Sebastian Hahn (t.hahn@hzdr.de )
 * 
 * @brief The View provides functionality for the DeepIterator. The first 
 * functionality is the construction of the DeepIterator type. The second part
 * of the functionality is providing the begin and end functions. Last but not 
 * least the view connects more than one layer.
 * 
 * 
 * We start with the first functionality, the construction of the DeepIterator.
 * The DeepIterator has several template parameter. For the most of that we had
 * written some instances. Most of these require template Parameter to work. The 
 * View build the types for navigator, accesssor and so on.
 * One design goal is a easy to use interface. From the container of the stl you 
 * known that all of them has the functions begin and end. The View gives you 
 * these two functions, I.e. you can use it, like a stl container.
 * The last functionality is, the view provides a parameter to picture nested
 * datastructres. This is down with the child template.
 * @tparam TContainer  This one describes the container, over wich elements you 
 * would like to iterate. This Templeate need has some Conditions: I. The Trait 
 * \b IsIndexable need a shape for TContainer. This traits, says wheter 
 * TContainer is array like (has []-operator overloaded) or list like; II. The
 * trait \b ComponentType has a specialication for TContainer. This trait gives the type
 * of the components of TContainer; III. The Funktion \b NeedRuntimeSize<TContainer>
 * need to be specified. For more details see NeedRuntimeSize.hpp ComponentType.hpp IsIndexable.hpp
 * @tparam TDirection The direction of the iteration. There are to posibilities
 * Forward and Backward. For more details see Direction.
 * @tparam TCollective is used to determine the collective properties of your 
 * iterator.
 * @tparam TChild The child is used to describe nested structures.
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
 */

#pragma once
#include "DeepIterator.hpp"
#include "Definitions/forward.hpp"
#include "Iterator/Concept.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include "Iterator/Policies.hpp"
#include "Traits/NumberElements.hpp"
#include "Traits/Componenttype.hpp"
#include "Definitions/hdinline.hpp"
#include <type_traits>
namespace hzdr 
{



template<
    typename TContainer,
    typename ComponentType,
    typename TAccessor,
    typename TNavigator,
    typename TChild>
struct View
{
public:
    typedef TContainer ContainerType;
    typedef ContainerType* ContainerPtr;
    typedef ContainerType& ContainerRef;
    
    typedef ComponentType*                                              ComponentPtr;
    typedef ComponentType&                                              ComponentRef;
    
    typedef TAccessor AccessorType;
    typedef TNavigator NavigatorType;
    typedef TChild ChildType;
    
    typedef DeepIterator<
        ContainerType,
        AccessorType,
        NavigatorType,
        ChildType> IteratorType;
        
    HDINLINE View() = default;
    HDINLINE View(View const &) = default;
    HDINLINE View(View &&) = default;
    
    template<
        typename TAccessor_,
        typename TNavigator_,
        typename TChild_>
    HDINLINE
    View(
         ContainerType& container,
         TAccessor_ && accessor,
         TNavigator_ && navigator,
         TChild_ && child
        ):
        containerPtr(&container),
        accessor(hzdr::forward<TAccessor_>(accessor)),
        navigator(hzdr::forward<TNavigator_>(navigator)),
        child(hzdr::forward<TChild_>(child))
    {}
        
    HDINLINE
    IteratorType
    end()
    {
        return IteratorType(containerPtr, accessor, navigator, child, details::constructorType::end());
    }
    
    HDINLINE
    IteratorType 
    rbegin()
    {
        return IteratorType(containerPtr, accessor, navigator, child, details::constructorType::rbegin());
    }
    
    HDINLINE
    IteratorType 
    rend()
    {
        return IteratorType(containerPtr, accessor, navigator, child, details::constructorType::rend());
    }
        
    HDINLINE
    IteratorType
    begin()
    {
        return IteratorType(containerPtr, accessor, navigator, child, details::constructorType::begin());
    }



protected:
    ContainerPtr containerPtr;
    AccessorType accessor;
    NavigatorType navigator;
    ChildType child;
};

template<
    typename TContainer,
    typename TConcept,
    typename ContainerNoRef = typename std::decay<TContainer>::type,
    typename ComponentType = typename traits::ComponentType<ContainerNoRef>::type>
auto 
HDINLINE
makeView(
    TContainer && con, 
    TConcept && concept)
->
    View<
        ContainerNoRef,
        ComponentType,
        decltype(details::makeAccessor<ContainerNoRef>(hzdr::forward<TConcept>(concept).accessor)),
        decltype(details::makeNavigator<ContainerNoRef>(hzdr::forward<TConcept>(concept).navigator)),
        decltype(details::makeIterator<ComponentType>(hzdr::forward<TConcept>(concept).child))>
{

        
        typedef ContainerNoRef                          ContainerType;

        typedef decltype(details::makeAccessor<ContainerNoRef>( hzdr::forward<TConcept>(concept).accessor)) AccessorType;
        typedef decltype(details::makeNavigator<ContainerNoRef>( hzdr::forward<TConcept>(concept).navigator)) NavigatorType;
        typedef decltype(details::makeIterator<ComponentType>(hzdr::forward<TConcept>(concept).child)) ChildType;
        typedef View<
            ContainerType,
            ComponentType,
            AccessorType,
            NavigatorType,
            ChildType> ResultType;
     //   hzdr::forward<TConcept>(concept).navigator.T();
        
        return ResultType(
            hzdr::forward<TContainer>(con), 
            details::makeAccessor<ContainerNoRef>(hzdr::forward<TConcept>(concept).accessor), 
            details::makeNavigator<ContainerNoRef>(hzdr::forward<TConcept>(concept).navigator), 
            details::makeIterator<ComponentType>(hzdr::forward<TConcept>(concept).child));
}

} // namespace hzdr
