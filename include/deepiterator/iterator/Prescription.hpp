#pragma once
#include "deepiterator/definitions/forward.hpp"
/**
 * @author Sebastian Hahn t.hahn <at> hzdr.de
 * @brief A concept consists of an accessor, a navigator and a child. A concept
 * decribes an abstract way to iterate through the data. The navigator and the 
 * accessor are not bound to a container.
 * 
 */

namespace hzdr 
{

namespace details 
{
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
    IteratorPrescription(TAccessor_ && acc,
                    TNavigator_ && navi):
        child(hzdr::NoChild()),
        navigator(hzdr::forward<TNavigator_>(navi)),
        accessor(hzdr::forward<TAccessor_>(acc))
    {}
    
    template<
        typename TNavigator_,
        typename TAccessor_,
        typename TChild_>
    HDINLINE
    IteratorPrescription(TAccessor_ && acc,
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
    const bool valgrind_debug = true;
} ;

template<typename Prescription>
struct PrescriptionTypes
{
    typedef typename Prescription::AccessorType AccessorType;
    typedef typename Prescription::NavigatorType NavigatorType;
    typedef typename Prescription::AccessorType ChildType;
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
    typename TNavigator>
HDINLINE
auto
makeIteratorPrescription(TAccessor&& accessor,
             TNavigator&& navigator)
-> hzdr::details::IteratorPrescription<
    typename std::decay<TAccessor>::type,
    typename std::decay<TNavigator>::type,
    hzdr::NoChild>
{
    
    typedef hzdr::details::IteratorPrescription< 
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        hzdr::NoChild> Iterator;
    
    return Iterator(
        hzdr::forward<TAccessor>(accessor), 
        hzdr::forward<TNavigator>(navigator));
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
    typename TChild>
HDINLINE
auto
makeIteratorPrescription(TAccessor && accessor,
             TNavigator && navigator,
             TChild && child
            )
-> hzdr::details::IteratorPrescription<
    typename std::decay<TAccessor>::type,
    typename std::decay<TNavigator>::type,
    typename std::decay<TChild>::type>
{
    
    typedef hzdr::details::IteratorPrescription< 
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        typename std::decay<TChild>::type> Iterator;
    
    return Iterator(
        hzdr::forward<TAccessor>(accessor), 
        hzdr::forward<TNavigator>(navigator),
        hzdr::forward<TChild>(child));
}

} // namespace hzdr

template<
    typename TAccessor,
    typename TNavigator,
    typename TChild
>
std::ostream& operator<<( std::ostream& out, hzdr::details::IteratorPrescription<TAccessor, TNavigator, TChild>  const & prescription) 
{
//      out << "Navigator: " << prescription.navigator << std::endl;
//     out << "Child: " << prescription.child << std::endl;
    return out;
}
