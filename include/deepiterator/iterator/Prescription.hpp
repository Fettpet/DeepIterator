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

#include "deepiterator/definitions/forward.hpp"
#include "deepiterator/definitions/hdinline.hpp"


namespace deepiterator 
{

namespace details 
{
/**
 * @author Sebastian Hahn t.hahn <at> deepiterator.de
 * @brief A Prescription consists of an accessor, a navigator and a child. 
 * A Prescription decribes an abstract way to iterate through the data. The 
 * navigator and the accessor are not bound to a container.
 * 
 */
template<
    typename TAccessor,
    typename TNavigator,
    typename TChild
>
struct IteratorPrescription
{
    typedef TNavigator NavigatorType;
    typedef TAccessor AccessorType;
    typedef TChild ChildType;
    
    static const bool hasChild = not std::is_same<
        TChild,
        deepiterator::NoChild
    >::value;
    
    HDINLINE 
    IteratorPrescription() = delete;
    
    HDINLINE 
    IteratorPrescription(IteratorPrescription const &) = default;
    
    HDINLINE 
    IteratorPrescription(IteratorPrescription &&) = default;
    
    template<
        typename TNavigator_,
        typename TAccessor_>
    HDINLINE
    IteratorPrescription(
            TAccessor_ && acc,
            TNavigator_ && navi
    ):
        child(deepiterator::NoChild()),
        navigator(deepiterator::forward<TNavigator_>(navi)),
        accessor(deepiterator::forward<TAccessor_>(acc))
    {}
    
    template<
        typename TNavigator_,
        typename TAccessor_,
        typename TChild_>
    HDINLINE
    IteratorPrescription(
            TAccessor_ && acc,
            TNavigator_ && navi,
            TChild_ && child
    ):
        child(deepiterator::forward<TChild_>(child)),
        navigator(deepiterator::forward<TNavigator_>(navi)),
        accessor(deepiterator::forward<TAccessor_>(acc))
    {}
    
    ChildType child;
    NavigatorType navigator;
    AccessorType accessor;
} ;


template<typename Prescription>
struct PrescriptionTypes
{
    using AccessorType = typename Prescription::AccessorType;
    using NavigatorType = typename Prescription::NavigatorType ;
    using ChildType = typename Prescription::ChildType;
};

} // namespace details


/**
 * @brief creates an iterator concept. This concept has no childs.
 * @param accessor Describes a concept to dereference the element 
 * @param navigator describe a concept of walking through the container
 * @return An iterator concept
 */
template<
    typename TAccessor,
    typename TNavigator
    >
HDINLINE
auto
makeIteratorPrescription(
        TAccessor&& accessor,
        TNavigator&& navigator
)
-> 
    deepiterator::details::IteratorPrescription<
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        deepiterator::NoChild
    >
{
    
    using Iterator = deepiterator::details::IteratorPrescription< 
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        deepiterator::NoChild
    >;
    
    return Iterator(
        deepiterator::forward<TAccessor>(accessor), 
        deepiterator::forward<TNavigator>(navigator)
    );
}

/**
 * @brief creates an iterator concept. This concept has no childs.
 * @param accessor Describes a concept to dereference the element 
 * @param navigator describe a concept of walking through the container
 * @return An iterator concept
 */
template<
    typename TAccessor,
    typename TNavigator,
    typename TChild
>
HDINLINE
auto
makeIteratorPrescription(
    TAccessor && accessor,
    TNavigator && navigator,
    TChild && child
)
-> 
deepiterator::details::IteratorPrescription<
    typename std::decay<TAccessor>::type,
    typename std::decay<TNavigator>::type,
    typename std::decay<TChild>::type
>
{
    
    using Prescription = deepiterator::details::IteratorPrescription< 
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        typename std::decay<TChild>::type
    >;
    
    return Prescription(
        deepiterator::forward<TAccessor>(accessor), 
        deepiterator::forward<TNavigator>(navigator),
        deepiterator::forward<TChild>(child)
    );
}

namespace details 
{

template<
    typename TContainer,
    typename TPrescription,
    typename TPrescriptionNoRef = typename std::decay<TPrescription>::type,
    typename = typename std::enable_if<not TPrescriptionNoRef::hasChild>::type
>
HDINLINE
auto
makeIteratorPrescription(
    TPrescription && prescription
)
-> 
    deepiterator::details::IteratorPrescription<
        decltype(makeAccessor<TContainer>(
            deepiterator::forward<TPrescription>(prescription).accessor)),
        decltype(makeNavigator<TContainer>(
            deepiterator::forward<TPrescription>(prescription).navigator)),
        deepiterator::NoChild
    >
{
    using Prescription = deepiterator::details::IteratorPrescription< 
        decltype(makeAccessor<TContainer>(
            deepiterator::forward<TPrescription>(prescription).accessor)),
        decltype(makeNavigator<TContainer>(
            deepiterator::forward<TPrescription>(prescription).navigator)),
        deepiterator::NoChild
    >;
    
    return Prescription(
        makeAccessor<TContainer>(
            deepiterator::forward<TPrescription>(prescription).accessor
        ), 
        makeNavigator<TContainer>(
            deepiterator::forward<TPrescription>(prescription).navigator
        )
    );
}


template<
    typename TContainer,
    typename ComponentType = typename deepiterator::traits::ComponentType<
        TContainer
    >::type,
    typename TPrescription,
    typename TPrescriptionNoRef = typename std::decay<TPrescription>::type,
    typename = typename std::enable_if<TPrescriptionNoRef::hasChild>::type
>
HDINLINE
auto
makeIteratorPrescription(
    TPrescription && prescription
)
-> 
    deepiterator::details::IteratorPrescription<
        decltype(makeAccessor<TContainer>(
            deepiterator::forward<TPrescription>(prescription).accessor)),
        decltype(makeNavigator<TContainer>(
            deepiterator::forward<TPrescription>(prescription).navigator)),
        decltype(makeIteratorPrescription<ComponentType>(
            deepiterator::forward<TPrescription>(prescription).child)
        )
    >
{

    using Prescription = deepiterator::details::IteratorPrescription< 
        decltype(makeAccessor<TContainer>(
            deepiterator::forward<TPrescription>(prescription).accessor)),
        decltype(makeNavigator<TContainer>(
            deepiterator::forward<TPrescription>(prescription).navigator)),
        decltype(makeIteratorPrescription<ComponentType>(
            deepiterator::forward<TPrescription>(prescription).child)
        )
    >;
    
    return Prescription(
        makeAccessor<TContainer>(
            deepiterator::forward<TPrescription>(prescription).accessor
        ), 
        makeNavigator<TContainer>(
            deepiterator::forward<TPrescription>(prescription).navigator
        ),
        makeIteratorPrescription<ComponentType>(
            deepiterator::forward<TPrescription>(prescription).child
        )
        
    );
    
}
  

    
} // namespace details

} // namespace deepiterator

template<
    typename TAccessor,
    typename TNavigator,
    typename TChild
>
std::ostream& operator<<( 
    std::ostream & out, 
    deepiterator::details::IteratorPrescription<
        TAccessor, 
        TNavigator, 
        TChild
    > const & prescription) 
{
//      out << "Navigator: " << prescription.navigator << std::endl;
//     out << "Child: " << prescription.child << std::endl;
    return out;
}
