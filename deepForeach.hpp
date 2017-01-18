#pragma once

#include "is_iteratorPair.hpp"
#include "isContainer.hpp"
#include <algorithm>
#include <iterator>

namespace detail
{
/**
 * Declaration of all templates
 * All for Iterators
 */
template<typename Iter,
         typename Functor>
void deepForeach(
                  Iter& begin,
                  Iter& end,
                 const Functor& functor,
                 std::true_type);
    
template<typename Iter,
         typename Functor>
void deepForeach(
                  Iter& begin,
                  Iter& end,
                 const Functor& functor,
                 std::false_type);

// All for Containers
template<typename TContainer,
         typename Functor>
void deepForeach(
                 TContainer& con,
                 const Functor& functor,
                 std::true_type,
                 std::true_type
                );

template<typename TContainer,
         typename Functor,
         typename TTrue,
         typename TFalse>
void deepForeach(
                 TContainer& con,
                 const Functor& functor,
                 std::true_type,
                 std::false_type
                );

template<typename TContainer,
         typename Functor>
void deepForeach(
                  TContainer& con,
                 const Functor& functor,
                 std::true_type
                );

template<typename TContainer,
         typename Functor>
void deepForeach(
                  TContainer& con,
                 const Functor& functor,
                 std::false_type
                );





    
template<typename Iter,
         typename Functor>
void deepForeach(
                  Iter& begin,
                  Iter& end,
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
                  Iter& begin,
                  Iter& end,
                 const Functor& functor,
                 std::false_type)
{
    std::for_each(begin, end, functor);
}




template<typename TContainer,
         typename Functor>
void deepForeach(
                 TContainer& con,
                 const Functor& functor,
                 std::true_type,
                 std::true_type
                )
{
    auto iter = con.begin();
        while(iter != con.end())
        {
            using bool_t = typename isContainer<decltype(*iter)>::type;
            deepForeach(*iter,
                        functor,
                        bool_t());
            iter++;
        }
}



template<typename TContainer,
         typename Functor>
void deepForeach(
                 TContainer& con,
                 const Functor& functor,
                 std::true_type,
                 std::false_type
                )
{
 
    
    std::for_each(con.begin(), con.end(), functor);
}

template<typename TContainer,
         typename Functor>
void deepForeach(
                  TContainer& con,
                 const Functor& functor,
                 std::true_type
                )
{
    using bool_tt = typename isContainer<decltype(*(con.begin()))>::type;
    

    deepForeach(con, functor, std::true_type(), bool_tt());    
    
}



template<typename TContainer,
         typename Functor>
void deepForeach(
                  TContainer& con,
                 const Functor& functor,
                 std::false_type
                )
{
    std::for_each(con.begin(), con.end(), functor);
}







} // namespace detail


template<typename Iter,
         typename Functor>
void deepForeach( Iter& begin,
                  Iter& end,
                 const Functor& functor)
{
    detail::deepForeach(
        begin,
        end,
        functor,
        is_iteratorPair< typename std::iterator_traits<Iter>::value_type >());
}

template<typename TContainer,
         typename Functor>
void deepForeach(TContainer& con,
                 const Functor& functor)
{
    using bool_t = typename isContainer<TContainer>::type;
    detail::deepForeach(
        con,
        functor,
        bool_t());
}

