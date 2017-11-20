/**
 * \struct Accessor
 * 
 * @author Sebastian Hahn (t.hahn[at]hzdr.de) 
 * 
 * @brief The Accesor is used to describe the current position of the iterator. 
 * The Accessor use several functions: 
 * 1. TComponent& get(TContaier*, TIndex&): returns the current value at the 
 * iterator position
 * 2. bool equal(TContainer*, TIndex&, TContainer*, TIndex&): returns true,
 * if the iterator is at the same position
 * 3. bool ahead(TContainer*, TIndex&, TContainer*, TIndex&): returns true,
 * if the first iterator is ahead the second one
 * 4. bool behind(TContainer* , TIndex&, TContainer*, TIndex&): returns true,
 * if the first iterator is behind the second one
 * The functions ahead and behind are only needed, if the iterator is random 
 * accessable. 
 * To use the default Accessor you need to spezify one trait for each function:
 * 1. get: hzdr::traits::accessor::Get<TContainer, TComponent, TIndex>
 * 2. ahead: hzdr::traits::accessor::Ahead<TContainer, TComponent, TIndex>
 * 3. equal: hzdr::traits::accessor::Equal<TContainer, TComponent, TIndex>
 * 4. behind: hzdr::traits::accessor::Behind<TContainer, TComponent, TIndex>
 * We had implemented two defaults Accessor. The first is used for arraylike 
 * data structures, the second use doubly link list like datastructures.
 * @tparam TContainer The container over which you like to iterate. 
 * @tparam TComponent The type of the container component. 
 * @tparam TIndex Type of the index to get access to the value of the iterator 
 * position.
 * @tparam TContainerCategory Type of default access parameter
 */

#pragma once
#include "PIC/Frame.hpp"
#pragma once
#include "PIC/Supercell.hpp"
#include <iostream>
#include "Traits/Componenttype.hpp"
#include "Definitions/hdinline.hpp"
#include "Traits/NumberElements.hpp"
#include "Traits/IndexType.hpp"
#include "Iterator/Categorie/DoublyLinkListLike.hpp"
#include "Traits/Accessor/Ahead.hpp"
#include "Traits/Accessor/Behind.hpp"
#include "Traits/Accessor/Equal.hpp"
#include "Traits/Accessor/Get.hpp"
#include "Definitions/forward.hpp"
namespace hzdr
{
namespace details
{
template <typename T>
class UndefinedAhead
{
    typedef char one;
    typedef long two;

    template <typename C> static one test( typeof(&C::UNDEFINED) ) ;
    template <typename C> static two test(...);    

public:
    enum { value = sizeof(test<T>(0)) == sizeof(char) };
}; // class UndefinedAhead
} // namespace details
     
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TContainerCategory,
    typename TGet = hzdr::traits::accessor::Get<TContainer, TComponent, TIndex, TContainerCategory>,
    typename TAhead = hzdr::traits::accessor::Ahead<TContainer, TComponent, TIndex, TContainerCategory>,
    typename TEqual = hzdr::traits::accessor::Equal<TContainer, TComponent, TIndex, TContainerCategory>,
    typename TBehind = hzdr::traits::accessor::Behind<TContainer, TComponent, TIndex, TContainerCategory>,
    bool isRandomAccessable = not details::UndefinedAhead<TAhead>::value >
struct Accessor
{
    typedef TContainer                                              ContainerType;
    typedef ContainerType*                                          ContainerPtr;
    typedef TComponent                                              ComponentType;
    typedef ComponentType*                                          ComponentPtr;
    typedef ComponentType&                                          ComponentRef;
    typedef TContainerCategory                                      ContainerCategory;
    typedef TIndex                                                  IndexType;
    
    HDINLINE Accessor() = default;
    HDINLINE Accessor(Accessor const &) = default;
    HDINLINE Accessor(Accessor &&) = default;
    
    HDINLINE 
    ComponentRef
    get(ContainerPtr containerPtr,
        IndexType & idx)
    {
        return _get(containerPtr, idx);
    }
    
    HDINLINE
    bool 
    debug_Test()
    {
        return true;
    }
    
    
    HDINLINE
    bool
    equal(ContainerPtr const containerPtr1,
          IndexType const & index1,
          ContainerPtr const containerPtr2,
          IndexType const & index2)
    {
        return _equal(containerPtr1, index1, containerPtr2, index2);
    }
    
    
    template<bool T = isRandomAccessable>
    HDINLINE 
    bool
    greater(ContainerPtr const containerPtr1,
          IndexType const & index1,
          ContainerPtr const containerPtr2,
          IndexType const & index2,
          std::enable_if<T == true>* = nullptr)
    {
        return _ahead(containerPtr1, index1, containerPtr2, index2);
    }
    
    template<bool T = isRandomAccessable>
    HDINLINE 
    bool
    lesser(ContainerPtr const 
    containerPtr1,
          IndexType const & index1,
          ContainerPtr const containerPtr2,
          IndexType const & index2,
          std::enable_if<T == true>* = nullptr)
    {
        return _behind(containerPtr1, index1, containerPtr2, index2);
    }
    
    
    
     TGet _get;
     TAhead _ahead;
     TEqual _equal;
     TBehind _behind;
} ;

/**
 * @brief the accessor concept.
 */
template<>
struct Accessor<
    details::UndefinedType, 
    details::UndefinedType,
    details::UndefinedType,
    details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType>
{
    typedef details::UndefinedType ContainerType;
    
    HDINLINE Accessor() = default;
    HDINLINE Accessor(Accessor const &) = default;
    HDINLINE Accessor(Accessor &&) = default;
    
    void testAcc(){}
 
    // for debugging with valgrind needed
    const bool value = false;
} ;

namespace details 
{




template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TIndex = typename traits::IndexType<TContainerNoRef>::type,
    typename TContainerCategory = typename traits::ContainerCategory<typename std::decay<TContainer>::type>::type,
    typename TComponent = typename hzdr::traits::ComponentType<TContainerNoRef>::type,
    typename TGet = hzdr::traits::accessor::Get<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TAhead = hzdr::traits::accessor::Ahead<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TEqual = hzdr::traits::accessor::Equal<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TBehind = hzdr::traits::accessor::Behind<TContainerNoRef, TComponent, TIndex, TContainerCategory>, 
    typename TAccessor>
HDINLINE
auto 
makeAccessor(TAccessor&&)
-> hzdr::Accessor<
    TContainerNoRef,
    TComponent,
    TIndex,
    TContainerCategory,
    TGet,
    TAhead,
    TEqual,
    TBehind>
{
    typedef hzdr::Accessor<
        TContainerNoRef,
        TComponent,
        TIndex,
        TContainerCategory> AccessorType;
    auto && accessor = AccessorType();
    return accessor;
}

template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TIndex = typename traits::IndexType<TContainerNoRef>::type,
    typename TContainerCategory = typename traits::ContainerCategory<typename std::decay<TContainer>::type>::type,
    typename TComponent = typename hzdr::traits::ComponentType<TContainerNoRef>::type,
    typename TGet = hzdr::traits::accessor::Get<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TAhead = hzdr::traits::accessor::Ahead<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TEqual = hzdr::traits::accessor::Equal<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TBehind = hzdr::traits::accessor::Behind<TContainerNoRef, TComponent, TIndex, TContainerCategory> >
auto 
HDINLINE
makeAccessor()
-> hzdr::Accessor<
    TContainerNoRef,
    TComponent,
    TIndex,
    TContainerCategory,
    TGet,
    TAhead,
    TEqual,
    TBehind>
{
    typedef hzdr::Accessor<
        TContainerNoRef,
        TComponent,
        TIndex,
        TContainerCategory,
        TGet,
        TAhead,
        TEqual,
        TBehind>                                          ResultType;
        
    return ResultType();
}

} // namespace details


/**
 * @brief creates an accessor
 */
template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TIndex = typename traits::IndexType<TContainerNoRef>::type,
    typename TContainerCategory = typename traits::ContainerCategory<typename std::decay<TContainer>::type>::type,
    typename TComponent = typename hzdr::traits::ComponentType<TContainerNoRef>::type,
    typename TGet = hzdr::traits::accessor::Get<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TAhead = hzdr::traits::accessor::Ahead<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TEqual = hzdr::traits::accessor::Equal<TContainerNoRef, TComponent, TIndex, TContainerCategory>,
    typename TBehind = hzdr::traits::accessor::Behind<TContainerNoRef, TComponent, TIndex, TContainerCategory> >
    
auto 
HDINLINE
makeAccessor(TContainer&&)
-> hzdr::Accessor<
    TContainerNoRef,
    TComponent,
    TIndex,
    TContainerCategory,
    TGet,
    TAhead,
    TEqual,
    TBehind>
{
    typedef hzdr::Accessor<
        TContainerNoRef,
        TComponent,
        TIndex,
        TContainerCategory,
        TGet,
        TAhead,
        TEqual,
        TBehind>                                         ResultType;
        
    return ResultType();
}






/**
 * @brief creates an accessor concept
 */
auto 
HDINLINE
makeAccessor()
-> hzdr::Accessor<
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType,
    hzdr::details::UndefinedType>
{
    typedef hzdr::Accessor<
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType>                           ResultType;
    return ResultType();
}

template<
    typename TAccessor>
HDINLINE
auto
makeAccessor(TAccessor && accessor)
->
decltype(hzdr::forward<TAccessor>(accessor))
{
    return hzdr::forward<TAccessor>(accessor);
}

}// namespace hzdr
