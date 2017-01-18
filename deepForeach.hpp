#include "is_iteratorPair.hpp"

#include <algorithm>
#include <iterator>

namespace detail
{

template<typename Iter,
         typename Functor>
void deepForeach(
                 const Iter& begin,
                 const Iter& end,
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
                    is_iteratorPair< typename std::iterator_traits<Iter>::value_type >());
        iter++;
    }
}


/* hasChild = false */
template<typename Iter,
         typename Functor>
void deepForeach(
                 const Iter& begin,
                 const Iter& end,
                 const Functor& functor,
                 std::false_type)
{
    std::for_each(begin, end, functor);
}

} // namespace detail


template<typename Iter,
         typename Functor>
void deepForeach(const Iter& begin,
                 const Iter& end,
                 const Functor& functor)
{
    detail::deepForeach(
        begin,
        end,
        functor,
        is_iteratorPair< typename std::iterator_traits<Iter>::value_type >());
}
