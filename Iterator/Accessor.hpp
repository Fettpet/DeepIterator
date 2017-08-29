/**
 * \struct Accessor
 * 
 * @author Sebastian Hahn (t.hahn[at]hzdr.de) 
 * 
 * @brief The accessor handle the access to a value. It is a policy in the 
 * DeepIterator. 
 * 
 * We had a trait called IsIndexable. It has the condition that the datastructure 
 * has operator [] overloaded. If the condition
 * is satisfied, you doesnt need to implement an own Accessor. In other cases, 
 * you need to write one.
 * The accessor has a function 
 * <b>C* get( T*).</b> T is the input datatype and is the output datatype. You need to 
 * declare C as ReturnType. i.e 
 * <b>typedef C ReturnType;</b>
 * 
 * @tparam TContainer The container over which you like to iterate. 
 * 
 * @tparam SFIANE Used intern for sfiane.
 */

#pragma once
#include "PIC/Frame.hpp"
#pragma once
#include "PIC/Supercell.hpp"
#include <iostream>
#include <boost/core/ignore_unused.hpp>
#include "Definitions/hdinline.hpp"
#include "Traits/NumberElements.hpp"
#include "Traits/IsIndexable.hpp"
namespace hzdr
{
namespace details
{
struct UndefinedType;
} // namespace details
     
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TAccess>
struct Accessor
{
    typedef TContainer                  ContainerType;
    typedef ContainerType*              ContainerPtr;
    typedef TComponent                  ComponentType;
    typedef ComponentType*              ComponentenPtr;
    typedef ComponentType&              ComponentRef;
    typedef TIndex                      IndexType;
    typedef TAccess                     Access;
    
    HDINLINE 
    ComponentRef
    get(ContainerPtr containerPtr,
        ComponentPtr componentPtr,
        TIndex && index)
    {
        Access access;
        return access(
                containerPtr, 
                componentPtr,
                std::forward<IndexType>(index));
    }
}

template<
    typename TComponent,
    typename TIndex,
    typename TAccess>
struct Accessor<details::UndefinedType, TComponent, TIndex, TAccess>
{
    typedef details::UndefinedType ContainerType;
};





namespace details
{

// template<
//     typename TContainer>
// auto 
// HDINLINE
// makeAccessor()
// -> hzdr::Accessor<typename std::remove_reference<TContainer>::type >
// {
//     return hzdr::Accessor<typename std::remove_reference<TContainer>::type>();
// }   

} // details

template<
    typename TContainer>
auto 
HDINLINE
makeAccessor(TContainer&&)
-> hzdr::Accessor<
    typename std::decay<TContainer>::type,
    typename traits::ComponentType<typename std::decay<TContainer>::type>::type>::type,
    typename traits::Accessing<typename traits::ContainerCategory<std::decay<TContainer>::type>::type>::type,
    typename traits::IndexType<std::decay<TContainer>::type>::type     IndexType;
    >
{
    typedef typename std::decay<TContainer>::type               ContainerType;
    typedef typename traits::ComponentType<ContainerType>::type ComponentType; 
    typedef typename traits::ContainerCategory<ContainerType>::type ContainerCategory;
    typedef typename traits::Accessing<ContainerCategory>::type     AccessingType;
    typedef typename traits::IndexType<ContainerType>::type     IndexType;
    typedef hzdr::Accessor<
        ContainerType, 
        ComponentType, 
        AccessingType, 
        IndexType>                                              ResultType;
        
    return ResultType();
}


auto 
HDINLINE
makeAccessor()
-> hzdr::Accessor<
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType>
{
    typedef hzdr::Accessor<
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType>                           ResultType;
    return ResultType();
}
}// namespace hzdr
