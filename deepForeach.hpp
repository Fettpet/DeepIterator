#include "is_iteratorPair.hpp"

#include <algorithm>
#include <iterator>

namespace detail
{

template<typename Iterator,
         typename Functor>
void deepForeach(
                 const Iterator& begin,
                 const Iterator& end,
                 const Functor& functor,
                 std::true_type)
{
    auto iter = begin;
    while(iter != end)
    {
        auto iterPair = *iter;
        deepForeach(iterPair.first,
                    iterPair.second,
                    functor,
                    is_iteratorPair< typename std::iterator_traits<Iterator>::value_type >());
        iter++;
    }
}


template<typename Iterator,
         typename Functor>
void deepForeach(
                 const Iterator& begin,
                 const Iterator& end,
                 const Functor& functor,
                 std::false_type)
{
    std::for_each(begin, end, functor);
}


} // namespace detail


template<typename Iterator,
         typename Functor>
void deepForeach(const Iterator& begin,
                 const Iterator& end,
                 const Functor& functor)
{
    detail::deepForeach(
        begin,
        end,
        functor,
        is_iteratorPair< typename std::iterator_traits<Iterator>::value_type >());
}
