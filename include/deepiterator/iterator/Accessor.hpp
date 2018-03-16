/* Copyright 2018 Sebastian Hahn

 * This file is part of DeepIterator.
 *
 * DeepIterator is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DeepIterator is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <iostream>
#include <cassert>

#include "deepiterator/traits/Traits.hpp"
#include "deepiterator/definitions/hdinline.hpp"
#include "deepiterator/definitions/forward.hpp"


namespace hzdr
{
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
 * if the first iterator is ahead the second one. 
 * 4. bool behind(TContainer* , TIndex&, TContainer*, TIndex&): returns true,
 * if the first iterator is behind the second one.
 * The functions ahead and behind are only avail, if the iterator is random 
 * accessable. 
 * To use the default Accessor you need to spezify the following traits for 
 * each function:
 * 1. get: hzdr::traits::accessor::Get<
 *      TContainer, 
 *      TComponent, 
 *      TIndex, 
 *      TContainerCategory
 * >
 * 2. ahead: hzdr::traits::accessor::Ahead<
 *      TContainer, 
 *      TComponent, 
 *      TIndex, 
 *      TContainerCategory
 * >
 * 3. equal: hzdr::traits::accessor::Equal<
 *      TContainer, 
 *      TComponent, 
 *      TIndex,
 *      TContainerCategory
 * >
 * 4. behind: hzdr::traits::accessor::Behind<
 *      TContainer, 
 *      TComponent, 
 *      TIndex,
 *      TContainerCategory
 * >
 * @tparam TContainer The container over which you like to iterate. 
 * @tparam TComponent The type of the container component. 
 * @tparam TIndex Type of the index to get access to the value of the iterator 
 * position.
 * @tparam TContainerCategory Type of default access parameter
 * @tparam TGet Trait to define, how to get the first element 
 * @tparam TEqual Trait to define, when two iterators are at the same position
 * @tparam TAhead Trait to define, when the first iterator is ahead the second 
 * one. Only needed if the iterator is random accessable.
 * @tparam TBehind Trait to define, when the first iterator is behind the second
 * one. Only needed if the iteartor is random accessable.
 * @tparam isRandomAccessable true, if thecontainer is random accessable, false
 * otherwise
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    typename TContainerCategory,
    typename TGet,
    typename TAhead,
    typename TEqual,
    typename TBehind,
    bool isRandomAccessable
>
struct Accessor
{
    using ContainerType = TContainer;
    using ContainerPtr = ContainerType*;
    using ComponentType = TComponent;
    using ComponentPtr = ComponentType*;
    using ComponentRef = ComponentType&;
    using ContainerCategory = TContainerCategory;
    using IndexType = TIndex;
    
    HDINLINE Accessor() = default;
    HDINLINE Accessor(Accessor const &) = default;
    HDINLINE Accessor(Accessor &&) = default;
    
    HDINLINE Accessor& operator=(Accessor const &) = default;
    HDINLINE Accessor& operator=(Accessor &&) = default;
    
    
    /**
     * @brief this function dereference a container and a index to get the 
     * component
     * @param containerPtr pointer to a container
     * @param idx current position of the iterator
     * @return component at the idx position
     */
    HDINLINE 
    auto
    get(
        ContainerPtr containerPtr,
        IndexType & idx
    )
    ->
    ComponentRef
    {
        assert(containerPtr != nullptr);
        return _get(
            containerPtr, 
            idx
        );
    }
    
    
    /**
     * @brief this function compares two iterator positions. 
     * @param containerPtr1 pointer to the container of the first iterator
     * @param idx1 current position of the first iterator 
     * @param containerPtr2 pointer to the container of the second iterator
     * @param idx2 current position of the second iterator
     * @return true if both iterators are at the same position, false otherwise
     */
    HDINLINE
    auto
    equal(
        ContainerPtr const containerPtr1,
        IndexType const & index1,
        ContainerPtr const containerPtr2,
        IndexType const & index2
    )
    ->
    bool
    {
        assert(containerPtr1 != nullptr);
        assert(containerPtr2 != nullptr);
        return _equal(
            containerPtr1,
            index1, 
            containerPtr2, 
            index2
        );
    }
    
    
    /**
     * @brief this function compares two iterator positions. 
     * @param containerPtr1 pointer to the container of the first iterator
     * @param idx1 current position of the first iterator 
     * @param containerPtr2 pointer to the container of the second iterator
     * @param idx2 current position of the second iterator
     * @return true if both iterators are on the same container and the first 
     * index is ahead the second one.
     */
    template<bool T = isRandomAccessable>
    HDINLINE 
    auto
    greater(
        ContainerPtr const containerPtr1,
        IndexType const & index1,
        ContainerPtr const containerPtr2,
        IndexType const & index2
    )
    ->
    typename std::enable_if<T == true, bool>::type
    {
        assert(containerPtr1 != nullptr);
        assert(containerPtr2 != nullptr);
        return _ahead(
            containerPtr1, 
            index1, 
            containerPtr2, 
            index2
        );
    }
    
    
    /**
     * @brief this function compares two iterator positions. 
     * @param containerPtr1 pointer to the container of the first iterator
     * @param idx1 current position of the first iterator 
     * @param containerPtr2 pointer to the container of the second iterator
     * @param idx2 current position of the second iterator
     * @return true if both iterators are on the same container and the first 
     * index is behind the second one.
     */
    template<bool T = isRandomAccessable>
    HDINLINE 
    auto
    lesser(
        ContainerPtr const containerPtr1,
        IndexType const & index1,
        ContainerPtr const containerPtr2,
        IndexType const & index2
    )
    ->
    typename std::enable_if<T == true, bool>::type
    {
        assert(containerPtr1 != nullptr);
        assert(containerPtr2 != nullptr);
        return _behind(
            containerPtr1, 
            index1, 
            containerPtr2, 
            index2
        );
    }
    
    
    
     TGet _get;
     TAhead _ahead;
     TEqual _equal;
     TBehind _behind;
} ;


/**
 * @brief the accessor prescription. This is only a threshold
 */
template<bool isRandomAccessable>
struct Accessor<
    details::UndefinedType, 
    details::UndefinedType,
    details::UndefinedType,
    details::UndefinedType,
    details::UndefinedType,
    details::UndefinedType,
    details::UndefinedType,
    details::UndefinedType,
    isRandomAccessable
>
{
    HDINLINE Accessor() = default;
    HDINLINE Accessor(Accessor const &) = default;
    HDINLINE Accessor(Accessor &&) = default;
} ;

namespace details 
{


/**
 * @brief This function use an accessor prescription and container template to
 * create a accessor. To use this function several traits must be defined.
 * 1. IndexType
 * 2. ContainerCategory
 * 3. ComponentType
 * 4. Get
 * 5. Ahead
 * 6. Equal
 * 7. Behind
 * @param TAccessor The accessor prescription. It is only virtual and not needed
 * @tparam TContainer. Type of the container.
 * @return An accessor with the functionallity given by the traits.
 */
template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TContainerCategory = typename traits::ContainerCategory<
        TContainerNoRef
    >::type,
    typename TIndex = typename traits::IndexType<
        TContainerNoRef,
        TContainerCategory
    >::type,
    typename TComponent = typename hzdr::traits::ComponentType<
        TContainerNoRef
    >::type,
    typename TGet = hzdr::traits::accessor::Get<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,                                          
    typename TAhead = hzdr::traits::accessor::Ahead<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    typename TEqual = hzdr::traits::accessor::Equal<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    typename TBehind = hzdr::traits::accessor::Behind<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >, 
    typename TAccessor,
    bool isRandomAccessable = hzdr::traits::IsRandomAccessable<
        TContainerNoRef,
        TContainerCategory
    >::value
>
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
    TBehind,
    isRandomAccessable
>
{
    using AccessorType =  hzdr::Accessor<
        TContainerNoRef,
        TComponent,
        TIndex,
        TContainerCategory,
        TGet,
        TAhead,
        TEqual,
        TBehind,
        isRandomAccessable
    > ;
    auto && accessor = AccessorType();
    return accessor;
}


/**
 * @brief This function use a container template to
 * create a accessor. To use this function several traits must be defined.
 * 1. IndexType
 * 2. ContainerCategory
 * 3. ComponentType
 * 4. Get
 * 5. Ahead
 * 6. Equal
 * 7. Behind
 * @tparam TContainer. Type of the container.
 * @return An accessor with the functionallity given by the traits.
 */
template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TContainerCategory = typename traits::ContainerCategory<
        typename std::decay<TContainer>::type
    >::type,    
    typename TIndex = typename traits::IndexType<
        TContainerNoRef,
        TContainerCategory
    >::type,    
    typename TComponent = typename hzdr::traits::ComponentType<
        TContainerNoRef
    >::type,
    typename TGet = hzdr::traits::accessor::Get<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    typename TAhead = hzdr::traits::accessor::Ahead<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    typename TEqual = hzdr::traits::accessor::Equal<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    typename TBehind = hzdr::traits::accessor::Behind<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    bool isRandomAccessable = hzdr::traits::IsRandomAccessable<
        TContainer,
        TContainerCategory
    >::value
>
auto 
HDINLINE
makeAccessor()
-> 
hzdr::Accessor<
    TContainerNoRef,
    TComponent,
    TIndex,
    TContainerCategory,
    TGet,
    TAhead,
    TEqual,
    TBehind,
    isRandomAccessable
>
{
    using ResultType = hzdr::Accessor<
        TContainerNoRef,
        TComponent,
        TIndex,
        TContainerCategory,
        TGet,
        TAhead,
        TEqual,
        TBehind,
        isRandomAccessable
    >;
        
    return ResultType();
}

} // namespace details


/**
 * @brief This function use a container template to
 * create a accessor. To use this function several traits must be defined.
 * 1. IndexType
 * 2. ContainerCategory
 * 3. ComponentType
 * 4. Get
 * 5. Ahead
 * 6. Equal
 * 7. Behind
 * @param TContainer. The container
 * @return An accessor with the functionallity given by the traits.
 */
template<
    typename TContainer,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename TContainerCategory = typename traits::ContainerCategory<
        typename std::decay<TContainer>::type
    >::type,
    typename TIndex = typename traits::IndexType<
        TContainerNoRef,
        TContainerCategory
    >::type,
    
    typename TComponent = typename hzdr::traits::ComponentType<
        TContainerNoRef
    >::type,
    typename TGet = hzdr::traits::accessor::Get<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,                                          
    typename TAhead = hzdr::traits::accessor::Ahead<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    typename TEqual = hzdr::traits::accessor::Equal<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    typename TBehind = hzdr::traits::accessor::Behind<
        TContainerNoRef, 
        TComponent, 
        TIndex, 
        TContainerCategory
    >,
    bool isRandomAccessable = hzdr::traits::IsRandomAccessable<
        TContainer,
        TContainerCategory
    >::value
>    
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
    TBehind,
    isRandomAccessable
>
{
    using ResultType = hzdr::Accessor<
        TContainerNoRef,
        TComponent,
        TIndex,
        TContainerCategory,
        TGet,
        TAhead,
        TEqual,
        TBehind,
        isRandomAccessable
    >;
        
    return ResultType();
}


/**
 * @brief creates an accessor prescription.
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
    hzdr::details::UndefinedType,
    false>
{
    using ResultType = hzdr::Accessor<
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        hzdr::details::UndefinedType,
        false
    >;
    return ResultType();
}


/**
 * @brief This function forwards an own accessor. This is a identity function
 * @param Accessor The accessor you like to forward
 * @return the accessor given by as parameter.
 */
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
