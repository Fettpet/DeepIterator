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


namespace hzdr 
{

namespace details 
{
/**
 * @author Sebastian Hahn t.hahn <at> hzdr.de
 * @brief A Prescription consists of an accessor, a navigator and a child. 
 * A Prescription decribes an abstract way to iterate through the data. The 
 * navigator and the accessor are not bound to a container.
 * 
 */
template<
    typename TAccessor,
    typename TNavigator,
    typename TChild>
struct IteratorPrescription
{
    typedef TNavigator NavigatorType;
    typedef TAccessor AccessorType;
    typedef TChild ChildType;
    
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
        child(hzdr::NoChild()),
        navigator(hzdr::forward<TNavigator_>(navi)),
        accessor(hzdr::forward<TAccessor_>(acc))
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
        child(hzdr::forward<TChild_>(child)),
        navigator(hzdr::forward<TNavigator_>(navi)),
        accessor(hzdr::forward<TAccessor_>(acc))
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
    hzdr::details::IteratorPrescription<
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        hzdr::NoChild
    >
{
    
    using Iterator = hzdr::details::IteratorPrescription< 
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        hzdr::NoChild
    >;
    
    return Iterator(
        hzdr::forward<TAccessor>(accessor), 
        hzdr::forward<TNavigator>(navigator)
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
hzdr::details::IteratorPrescription<
    typename std::decay<TAccessor>::type,
    typename std::decay<TNavigator>::type,
    typename std::decay<TChild>::type
>
{
    
    using Iterator = hzdr::details::IteratorPrescription< 
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        typename std::decay<TChild>::type
    >;
    
    return Iterator(
        hzdr::forward<TAccessor>(accessor), 
        hzdr::forward<TNavigator>(navigator),
        hzdr::forward<TChild>(child)
    );
}

} // namespace hzdr

template<
    typename TAccessor,
    typename TNavigator,
    typename TChild
>
std::ostream& operator<<( 
    std::ostream & out, 
    hzdr::details::IteratorPrescription<
        TAccessor, 
        TNavigator, 
        TChild
    > const & prescription) 
{
//      out << "Navigator: " << prescription.navigator << std::endl;
//     out << "Child: " << prescription.child << std::endl;
    return out;
}
