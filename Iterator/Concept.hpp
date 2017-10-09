#pragma once

/**
 * @author Sebastian Hahn t.hahn <at> hzdr.de
 * @brief A concept consists of an accessor, a navigator and a child. A concept
 * decribes an abstract way to iterate through the data. The navigator and the 
 * accessor are not bound to a container.
 * 
 */

namespace hzdr 
{
struct NoChild;
namespace details 
{
template<
    typename TAccessor,
    typename TNavigator,
    typename TChild>
struct IteratorConcept
{
    typedef TNavigator NavigatorType;
    typedef TAccessor AccessorType;
    typedef TChild ChildType;
    
    template<
        typename TNavigator_,
        typename TAccessor_>
    IteratorConcept(TAccessor_ && acc,
                    TNavigator_ && navi):
        child(hzdr::NoChild()),
        navigator(std::forward<TNavigator_>(navi)),
        accessor(std::forward<TAccessor_>(acc))
    {}
    
    template<
        typename TNavigator_,
        typename TAccessor_,
        typename TChild_>
    IteratorConcept(TAccessor_ && acc,
                    TNavigator_ && navi,
                    TChild_ && child
                   ):
        child(std::forward<TChild_>(child)),
        navigator(std::forward<TNavigator_>(navi)),
        accessor(std::forward<TAccessor_>(acc))
    {}
    
    ChildType child;
    NavigatorType navigator;
    AccessorType accessor;
    
};

template<typename Concept>
struct ConceptTypes
{
    typedef typename Concept::AccessorType AccessorType;
    typedef typename Concept::NavigatorType NavigatorType;
    typedef typename Concept::AccessorType ChildType;
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
makeIteratorConcept(TAccessor&& accessor,
             TNavigator&& navigator)
-> hzdr::details::IteratorConcept<
    typename std::decay<TAccessor>::type,
    typename std::decay<TNavigator>::type,
    hzdr::NoChild>
{
    
    typedef hzdr::details::IteratorConcept< 
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        hzdr::NoChild> Iterator;
    
    return Iterator(
        std::forward<TAccessor>(accessor), 
        std::forward<TNavigator>(navigator));
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
makeIteratorConcept(TAccessor && accessor,
             TNavigator && navigator,
             TChild && child
            )
-> hzdr::details::IteratorConcept<
    typename std::decay<TAccessor>::type,
    typename std::decay<TNavigator>::type,
    typename std::decay<TChild>::type>
{
    
    typedef hzdr::details::IteratorConcept< 
        typename std::decay<TAccessor>::type,
        typename std::decay<TNavigator>::type,
        typename std::decay<TChild>::type> Iterator;
    
    return Iterator(
        std::forward<TAccessor>(accessor), 
        std::forward<TNavigator>(navigator),
        std::forward<TChild>(child));
}

} // namespace hzdr
