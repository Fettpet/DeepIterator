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


#include "deepiterator/definitions/hdinline.hpp"
#include "deepiterator/definitions/forward.hpp"

#include "deepiterator/DeepIterator.hpp"

#include "deepiterator/iterator/Prescription.hpp"
#include "deepiterator/iterator/Accessor.hpp"
#include "deepiterator/iterator/Navigator.hpp"
#include "deepiterator/iterator/SliceNavigator.hpp"
#include "deepiterator/traits/Traits.hpp"

#include <type_traits>
namespace hzdr 
{



/**
 * \struct View
 * @author Sebastian Hahn (t.hahn@hzdr.de )
 * 
 * @brief The View is a composition of a Prescription and a datastructure. It is
 * used to generate the DeepIterator. The View has four ways to create a Deep -
 * Iterator:
 * 1. begin()
 * 2. end()
 * 3. rbegin()
 * 4. rend()
 * 
 * @tparam TContainer : This one describes the container, over whose elements 
 * you would like to iterate. This template need the trait \b ComponentType has 
 * a specialication for TContainer. This trait gives the type of the components 
 * of TContainer; \see Componenttype.hpp 
 * @tparam TComponent Component type of the container.
 * @tparam TAccessor The accessor descripe the access to and position of the 
 * components of TContainer. \see Accessor.hpp
   @tparam TNavigator The navigator describe the way to walk through the data. 
   It describe the first element, the next element and the after last element.
   \see Navigator.hpp
 
   @tparam TChild The child is the template parameter to realize nested 
   structures.  This template has several 
   requirements: 
    1. it need to spezify an Iterator type. These type need operator++, 
        operator*, operator=, operator== and a default constructor.
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
   @tparam TIndexType Type of the index. The index is used to access the 
    component within the container. The index must support a cast from int 
    especially from 0.
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
 */
template<
    typename TContainer,
    typename TComponentType,
    typename TAccessor,
    typename TNavigator,
    typename TChild,
    typename TIndexType,
    bool hasConstantSize,
    bool isBidirectional,
    bool isRandomAccessable>
struct View
{
public:
    using ContainerType = TContainer;
    using ContainerPtr = ContainerType*;
    using ContainerRef = ContainerType&;
    
    using ComponentType = TComponentType;
    using ComponentPtr = ComponentType*;
    using ComponentRef = ComponentType&;
    
    using AccessorType = TAccessor;
    using NavigatorType = TNavigator;
    using ChildType = TChild;
    
    using IteratorType = DeepIterator<
        ContainerType,
        AccessorType,
        NavigatorType,
        ChildType,
        TIndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable
    >;
        
    HDINLINE View() = default;
    HDINLINE View(View const &) = default;
    HDINLINE View(View &&) = default;
    HDINLINE View& operator=(View const &) = default;
    HDINLINE View& operator=(View &&) = default;
    
    /**
     * @brief This is the constructor to create a useable view.
     * @param container The container over which you like to iterate
     * @param accessor Define the way how we access the data within the container
     * @param navigator define the way how the iterator goes through the container
     * @param child other iterator to handle nested datastructures
     */
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
    {
        static_assert(std::is_same<
            typename std::decay<TAccessor_>::type,
            typename std::decay<TAccessor>::type>::value,
            "The type of the accessor given by the template and the accessor given as parameter are not the same"
        );
        static_assert(std::is_same<
            typename std::decay<TNavigator_>::type,
            typename std::decay<TNavigator>::type>::value,
            "The type of the accessor given by the template and the accessor given as parameter are not the same"
        );
    }

    
    /**
    * @brief This function creates an iterator, which is at the after-last-element
    */
    HDINLINE
    auto
    end()
    -> 
    IteratorType
    {
        return IteratorType(
            containerPtr, 
            accessor, 
            navigator, 
            child, 
            details::constructorType::end()
        );
    }
    
    
    /**
    * @brief This function creates an iterator, which is at the last element
    */
    template<bool T = isBidirectional>
    HDINLINE
    auto 
    rbegin()
    -> 
    typename std::enable_if<
        T == true, 
        IteratorType
    >::type
    {
        return IteratorType(
            containerPtr, 
            accessor, 
            navigator, 
            child, 
            details::constructorType::rbegin()
        );
    }
    
    
    /**
    * @brief This function creates an iterator, which is at the before-first-
    * element
    */
    template<bool T = isBidirectional>
    HDINLINE
    auto
    rend()
    -> 
    typename std::enable_if<
        T == true,
        IteratorType
    >::type
    {
        return IteratorType(
            containerPtr, 
            accessor, 
            navigator, 
            child, 
            details::constructorType::rend()
        );
    }
    
    
    /**
    * @brief This function creates an iterator, which is at the first element
    */
    HDINLINE
    IteratorType
    begin()
    {
        return IteratorType(
            containerPtr, 
            accessor, 
            navigator, 
            child, 
            details::constructorType::begin()
        );
    }



protected:
    ContainerPtr containerPtr;
    AccessorType accessor;
    NavigatorType navigator;
    ChildType child;
} ;

/**
 * @brief Use a container and a prescription to create a view.
 * @param TContainer the container, over which you like to iterate
 * @param TPrescription the prescription of the layers
 * @return a view
 */
template<
    typename TContainer,
    typename TPrescription,
    typename TContainerNoRef = typename std::decay<TContainer>::type,
    typename ComponentType = typename traits::ComponentType<
        TContainerNoRef
    >::type,
    typename ContainerCategoryType = typename traits::ContainerCategory<
        TContainerNoRef
    >::type,    
    typename IndexType = typename hzdr::traits::IndexType<
        TContainerNoRef,
        ContainerCategoryType
    >::type,
    bool hasConstantSize = traits::HasConstantSize<TContainerNoRef>::value,
    bool isBidirectional = hzdr::traits::IsBidirectional<
        TContainerNoRef, 
        ContainerCategoryType
    >::value,
    bool isRandomAccessable = hzdr::traits::IsRandomAccessable<
        TContainerNoRef, 
        ContainerCategoryType
    >::value
>
auto 
HDINLINE
makeView(
    TContainer && con, 
    TPrescription && concept
)
->
    View<
        TContainerNoRef,
        ComponentType,
        decltype(details::makeAccessor<TContainerNoRef>(
            hzdr::forward<TPrescription>(concept).accessor
        )),
        decltype(details::makeNavigator<TContainerNoRef>(
            hzdr::forward<TPrescription>(concept).navigator
        )),
        decltype(details::makeIterator<ComponentType>(
            hzdr::forward<TPrescription>(concept).child
        )),
        IndexType,
        hasConstantSize,
        isBidirectional,
        isRandomAccessable
    >
{

        
        using ContainerType = TContainerNoRef;

        using AccessorType = decltype(details::makeAccessor<ContainerType>( 
            hzdr::forward<TPrescription>(concept).accessor
        ));
        using NavigatorType = decltype(details::makeNavigator<ContainerType>( 
            hzdr::forward<TPrescription>(concept).navigator
        ));
        using ChildType = decltype(details::makeIterator<ComponentType>(
            hzdr::forward<TPrescription>(concept).child
        ));
        using ResultType = View<
            ContainerType,
            ComponentType,
            AccessorType,
            NavigatorType,
            ChildType,
            IndexType,
            hasConstantSize,
            isBidirectional,
            isRandomAccessable
        >;
     

        auto && accessor = details::makeAccessor<ContainerType>(
            concept.accessor
        );
        
        auto && navigator = details::makeNavigator<ContainerType>(
            concept.navigator
        );
        auto && child = details::makeIterator<ComponentType>(
            concept.child
        );
        auto && result =  ResultType(
            hzdr::forward<TContainer>(con), 
            accessor, 
            navigator, 
            child
        );
        return result;
}

} // namespace hzdr
