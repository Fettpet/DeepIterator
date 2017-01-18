#include <iterator>
#include <type_traits>


template <typename T>
struct is_iteratorPair_trait
{
    template<
        typename TT,
        typename Dummy = typename std::iterator_traits<
            typename TT::first_type
        >::iterator_category
    >
    static std::true_type test(int);

    template <typename >
    static std::false_type test(...);

    using type= decltype(test<T>(0));
    static constexpr const bool value= type::value;
};

template <typename T>
struct is_iteratorPair
  : is_iteratorPair_trait<T>::type
{};
