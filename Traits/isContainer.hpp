#pragma once
#include <type_traits>

namespace Trait 
{
template <typename T>
struct isContainer_trait
{
    typedef typename std::remove_reference<T>::type TUnref;
    template<
        typename TT,
        typename Dummy = typename TT::iterator
    >
    static std::true_type test(int);

    template <typename >
    static std::false_type test(...);

    using type= decltype(test<TUnref>(0));
    static constexpr const bool value= type::value;
};

template <typename T>
struct isContainer
  : isContainer_trait<T>::type
{};

}// Trait