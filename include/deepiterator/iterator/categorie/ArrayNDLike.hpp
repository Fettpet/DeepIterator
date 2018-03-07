#pragma once
#include "deepiterator/traits/Traits.hpp"
#include "deepiterator/definitions/hdinline.hpp"

namespace hzdr
{
namespace container
{
namespace categorie
{
/**
This is the categorie of the nd array. It need the dimension n
@tparam n The dimension of the array.
*/
template<unsigned int n>
struct ArrayNDLike;



} //namespace container

} // namespace Categorie

namespace detail 
{
    /**
     * @brief This function use a n-d array as index and a n-d array as container
     * size and calculate a linear index.
     * @param idx The index you like to transform into a linear index (int)
     * @param containerSize The size of each dimension
     * @return a linear index
     */
    template<
        uint_fast32_t Dim,
        typename TSize,
        typename TIdx,
        typename = typename std::enable_if< 
            not std::is_integral<TIdx>::value 
        >::type
    >
    auto 
    HDINLINE
    idxndToInt(
        TIdx const & idx,
        TSize const & containerSize 
    )
    -> int
    {
        int result{0};
        for(uint i=0; i<Dim; ++i)
        {
            int size=1;
            for(uint j=0; j<i; ++j)
            {
                assert(containerSize[j] != 0);
                size *= containerSize[j];
            }
            result += idx[i] * size;
        }
        return result;
    }
    
    
      /**
     * @brief This function is a helper function, if the idx is an integer
     * @param idx The index you like to transform into a linear index (int)
     * @param containerSize The size of each dimension
     * @return idx
    */
    template<
        uint_fast32_t Dim,
        typename TSize,
        typename TContainer,
        typename = typename std::enable_if< 
             std::is_integral<TSize>::value 
        >::type
    >
    auto 
    HDINLINE
    idxndToInt(
        TSize const & idx,
        TContainer const & 
    )
    -> int
    {
        return idx;
    }
     
    /**
     * @brief This function is a helper function, if the idx is an integer
     * @param idx The index you like to transform into a linear index (int)
     * @param containerSize The size of each dimension
     * @return idx
     */
    template<
        uint_fast32_t Dim,
        typename TSize
    >
    auto 
    HDINLINE 
    intToIdxnd(
        int const & idx,
        TSize const & containerSize
    )
    -> TSize
    {
        TSize result;
        auto remaining = idx;
        for(uint i=0u ;i<Dim -1; ++i)
        {
            int size = 1;
            for( uint j=0u; j<i; ++j)
            {
                size *= (containerSize[j] > 0) * containerSize[j];
            }
            result[i] = ((remaining + size -1) / size) % containerSize[i];
            remaining -= result[i] * size;
        }
        int size = 1;
        for( uint j=0u; j<Dim- 1; ++j)
        {
                size *= containerSize[j] + (containerSize[j] == 0);
        }
        result[Dim - 1] = ((remaining + size -1) / size);
        
        return result;
    }
    
}
// implementation of the traits
namespace traits
{
template<
    typename TContainer,
    unsigned int Dim
>
struct IsBidirectional<
    TContainer, 
    hzdr::container::categorie::ArrayNDLike<Dim> 
>
{
    static const bool value = true;
} ;    

template<
    typename TContainer,
    unsigned int Dim
>
struct IsRandomAccessable<
    TContainer, 
    hzdr::container::categorie::ArrayNDLike<Dim> 
>
{
    static const bool value = true;
} ;

namespace accessor 
{
/**
 * @brief Get the value out of the container at the current iterator position.
 * \see Get.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    unsigned int Dim
>
struct Get<
    TContainer, 
    TComponent, 
    TIndex, 
    hzdr::container::categorie::ArrayNDLike<Dim> 
>
{
    HDINLINE
    TComponent&
    operator() (
        TContainer* con, 
        TIndex & idx
    )
    {
        return (*con)[idx];
    }
    
} ;

/**
 * @brief check whether to iterators are at the same position. \see Equal.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    unsigned int Dim
>
struct Equal<
    TContainer, 
    TComponent, 
    TIndex, 
    hzdr::container::categorie::ArrayNDLike<Dim> 
>
{
    HDINLINE
    bool
    operator() (
        TContainer * const con1, 
        TIndex const & idx1, 
        TContainer * const con2, 
        TIndex const & idx2)
    {
        // is not implemented. Specify the trait
        return con1 == con2 && idx1 == idx2;
    }
    
} ;

/**
 * @brief check whether the first iterator is ahead the second one. 
 * \see Ahead.hpp
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    unsigned int Dim
>
struct Ahead<
    TContainer, 
    TComponent, 
    TIndex, 
    hzdr::container::categorie::ArrayNDLike<Dim> 
>
{
    HDINLINE
    bool
    operator() (
        TContainer * const con1, 
        TIndex const & idx1, 
        TContainer * const con2, 
        TIndex const & idx2
    )
    {
        using namespace hzdr::detail;
        return (idxndToInt<Dim>(
            idx1,
            con1->dim()) 
        > idxndToInt<Dim>(
            idx2,
            con2->dim()))
        && con1 == con2;
    }
    
} ;

/**
 * @brief check whether the first iterator is behind the first one. 
 */
template<
    typename TContainer,
    typename TComponent,
    typename TIndex,
    unsigned int Dim
>
struct Behind<
    TContainer, 
    TComponent, 
    TIndex, 
    hzdr::container::categorie::ArrayNDLike<Dim> 
>
{
    HDINLINE
    bool
    operator() (
        TContainer * const con1, 
        TIndex const & idx1, 
        TContainer * con2, 
        TIndex const & idx2)
    {
        using namespace hzdr::detail;
        return (idxndToInt<Dim>(
            idx1,
            con1->dim()) 
        < idxndToInt<Dim>(
            idx2,
            con2->dim()))
        && con1 == con2;
    }
    
} ;
    
} // namespace accessor
    
    
    
    
namespace navigator
{
/**
 * @brief implementation to get the first element within a container. For further
 * details \see FirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    unsigned int Dim
>
struct FirstElement<
    TContainer, 
    TIndex, 
    hzdr::container::categorie::ArrayNDLike<Dim> 
>
{
    HDINLINE
    void
    operator() (
        TContainer* con, 
        TIndex& idx
    )
    {
        using namespace hzdr::detail;
        idx = intToIdxnd<Dim>(
            0,
            con->dim()
        );
    }
    
} ;

/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange,
    unsigned int Dim
>
struct NextElement<
    TContainer,
    TIndex,
    TRange,
    hzdr::container::categorie::ArrayNDLike<Dim> >
{
    template<
        typename TContainerSize,
        typename TIndex_>
    HDINLINE
    TRange
    operator() (
        TContainer* container, 
        TIndex& idx, 
        TIndex_ const & range,
        TContainerSize& size)
    {
        using namespace hzdr::detail;
        auto newIdxInt = idxndToInt<Dim>(idx, container->dim())
                       + idxndToInt<Dim>(range, container->dim());


        if(newIdxInt >= size(container))
        {
            idx = container->dim();
            return (newIdxInt - (size(container)-1));
        }
        else 
        {
            idx = intToIdxnd<Dim>(
                newIdxInt,
                container->dim()
            );
            return 0;
        }
    }
    
} ;

/**
 * @brief Implementation to check whether the end is reached. For further 
 * informations \see AfterLastElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    unsigned int Dim
>
struct AfterLastElement<
    TContainer, 
    TIndex, 
    hzdr::container::categorie::ArrayNDLike<Dim> >
{
    template<typename TSizeFunction>
    HDINLINE
    bool
    test (
        TContainer* conPtr, 
        TIndex const & idx, 
        TSizeFunction const & size
    )
    const
    {
        using namespace hzdr::detail;
//         std::cout << "After last element test " << conPtr->dim() << " size " << idx << std::endl;
        return idxndToInt<Dim>(
                idx,
                conPtr->dim()
        ) >= size(conPtr);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set(TContainer* conPtr, TIndex & idx, TSizeFunction const & size)
    const
    {
        using namespace hzdr::detail;
        idx = intToIdxnd<Dim>(
            size(conPtr),
            conPtr->dim()
        );
    }
    
} ;

/**
 * @brief Implementation of the array like last element trait. For further details
 * \see LastElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    unsigned int Dim
>
struct LastElement<
    TContainer,
    TIndex,
    hzdr::container::categorie::ArrayNDLike<Dim> >
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (
        TContainer* conPtr, 
        TIndex& index, 
        TSizeFunction& size
    )
    {
        using namespace hzdr::detail;
        index = intToIdxnd<Dim>(
            size(conPtr) - 1,
            conPtr->dim()
        );
    }
    
} ;

/**
 * @brief The implementation to get the last element in a array like data
 * structure. For futher details \see PreviousElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TRange,
    unsigned int Dim
>
struct PreviousElement<
    TContainer,
    TIndex,
    TRange,
    hzdr::container::categorie::ArrayNDLike<Dim> >
{
    template<
        typename T,
        typename TRange_>
    HDINLINE
    TRange
    operator() (
        TContainer* container, 
        TIndex& idx, 
        TRange_ const & jumpsize,
        T const &
    )
    {
        using namespace hzdr::detail;
        
        auto const newIdxInt = idxndToInt<Dim>(
            idx,
            container->dim()
        ) 
        - 
        idxndToInt<Dim>(
            jumpsize,
            container->dim()
        );
        if(newIdxInt < 0)
        {
            idx = intToIdxnd<Dim>(
                static_cast<TRange>(-1),
                container->dim()
            );
            return static_cast<TRange>(-1) * newIdxInt;
        }
        else 
        {
            idx = intToIdxnd<Dim>(
                newIdxInt,
                container->dim()
            );
            return 0;
        }
    }
    
} ;

/**
 * @brief Implmentation to get check whether the iterator is on the element 
 * before the first one. \see BeforeFirstElement.hpp
 */
template<
    typename TContainer,
    typename TIndex,
    typename TOffset,
    unsigned int Dim
>
struct BeforeFirstElement<
    TContainer, 
    TIndex,
    TOffset,
    hzdr::container::categorie::ArrayNDLike<Dim> >
{
    template<typename TSizeFunction>
    HDINLINE
    bool
    test (
        TContainer* container, 
        TIndex const & idx, 
        TSizeFunction&)
    const
    {
        using namespace hzdr::detail;
        
        return idxndToInt<Dim>(
            idx, 
            container->dim()
        ) 
        < 0;
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set (
        TContainer* container, 
        TIndex & idx, 
        TSizeFunction&
    )
    const
    {
        using namespace hzdr::detail;
        idx = intToIdxnd<Dim>(
            -1,
            container->dim()
        );
    }
    
} ;

    
}// namespace navigator
} // namespace traits
    
}// namespace hzdr
