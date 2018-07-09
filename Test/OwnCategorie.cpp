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
 * along with DeepIterator.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#define BOOST_TEST_MODULE OwnCategorieBoostVector
#include <boost/test/included/unit_test.hpp>

#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"
#include <boost/container/vector.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
/**
 * @brief We use this test to give you an advice how you can create an own 
 * categorie. We like to use the map out of the std library as a categorie. The 
 * namespacing is like the namespacing in the iterator package. The first thing 
 * you need to do, give the categorie a name. 
 */
// this contains all includes for all classes we need.
#include "deepiterator/iterator/Categorie.hpp"
namespace deepiterator 
{
namespace container
{
namespace categorie
{
// We name our new categorie MapLike.
struct MapLike;
} //namespace container

} // namespace Categorie


/**
 * The next thing you need to do, specify the needed traits for accessor and 
 * navigator and the other needed:
 * 1. ComponentType
 * 1. IndexType,
 * 2. RangeType,
 * Four for the accessor Behaviour
 * 3. at
 * 4. equal
 * 5. ahead
 * 6. behind
 * Six for the navigator Behaviour
 * 7. firstElement
 * 8. nextElement
 * 9. EndElement
 * 10. lastElement 
 * 11. previousElement
 * 12. REndElement
 * We use the std iterator to iterate over the data struc
 */

namespace traits 
{

template<
    typename T,
    typename SFIANE>
struct RangeType<boost::container::vector<T>, SFIANE> 
{
    typedef int_fast32_t type; 
};

template<
    typename T,
    typename SFIANE>
struct IndexType<boost::container::vector<T>, SFIANE> 
{
    typedef int_fast32_t type; 
};

template<
    typename T>
struct ComponentType<boost::container::vector<T> > 
{
    typedef T type; 
};

template<
    typename T, 
    typename ContainerCategorie>
struct IsBidirectional<boost::container::vector<T>, ContainerCategorie> 
{
    static const bool value = true;
};

template<
    typename T,
    typename ContainerCategorie>
struct IsRandomAccessable<boost::container::vector<T>, ContainerCategorie > 
{
    static const bool value = true;
};

template<
    typename T>
struct HasConstantSize<boost::container::vector<T> > 
{
    static const bool value = false;
};

template< 
    typename T>
struct Size<boost::container::vector<T>>
{
    typedef boost::container::vector<T> Container;
    
    HDINLINE
    int_fast32_t 
    operator()(Container* container)
    const
    { 
        return container->size();
    }
};

namespace accessor 
{
/**
 * @brief Get the value out of the container at the current iterator position.
 * \see At.hpp
 */
template<
    typename TComponent,
    typename TCategorie,
    typename TIndex>
struct At<
    boost::container::vector<TComponent> , 
    TComponent, 
    TIndex, 
    TCategorie
    >
{
    typedef boost::container::vector<TComponent> TContainer;
    
    HDINLINE
    TComponent&
    operator() (
        TContainer* con,
        TComponent*&,
        TIndex const & idx)
    {
        // is not implemented. Specify the trait
        return (*con)[idx];
    }
};

/**
 * @brief check whether to iterators are at the same position. \see Equal.hpp
 */
template<
    typename TComponent,
    typename TCategorie,
    typename TIndex>
struct Equal<
    boost::container::vector<TComponent> , 
    TComponent, 
    TIndex, 
    TCategorie
    >
{
    typedef boost::container::vector<TComponent> TContainer;
    HDINLINE
    bool
    operator() (
        TContainer * const con1,
        TComponent* const &,
        TIndex const & idx1, 
        TContainer * const con2,
        TComponent* const &,
        TIndex const & idx2)
    {
        // is not implemented. Specify the trait
        return con1 == con2 && idx1 == idx2;
    }
};

/**
 * @brief check whether the first iterator is ahead the second one. 
 * \see Ahead.hpp
 */
template<
    typename TComponent,
    typename TCategorie,
    typename TIndex>
struct Ahead<
    boost::container::vector<TComponent> , 
    TComponent, 
    TIndex, 
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    
    HDINLINE
    bool
    operator() (
        TContainer * const con1,
        TComponent* const &,
        TIndex const & idx1, 
        TContainer * const con2,
        TComponent* const &,
        TIndex const & idx2)
    {
        // is not implemented. Specify the trait
        return idx1 > idx2 && con1 == con2;
    }
};

/**
 * @brief check whether the first iterator is behind the first one. 
 */
template<
    typename TComponent,
    typename TCategorie,
    typename TIndex>
struct Behind<
    boost::container::vector<TComponent> , 
    TComponent, 
    TIndex, 
    TCategorie
    >
{
    typedef boost::container::vector<TComponent> TContainer;
    
    
    HDINLINE
    bool
    operator() (
        TContainer * const con1,
        TComponent* const &,
        TIndex const & idx1, 
        TContainer * const con2,
        TComponent* const &,
        TIndex const & idx2)
    {
        // is not implemented. Specify the trait
        return con1 == con2 && idx1 < idx2 ;
    }
};
    
} // namespace accessor

    
namespace navigator
{
/**
 * @brief implementation to get the first element within a container. For further
 * details \see BeginElement.hpp
 */
template<
    typename TComponent,
    typename TIndex,
    typename TCategorie>
struct BeginElement<
    boost::container::vector<TComponent>,
    TComponent,
    TIndex, 
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    HDINLINE
    void
    operator() (
        TContainer*,
        TComponent*&,
        TIndex& idx)
    {
        idx = static_cast<TIndex>(0);
    }
};

/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TComponent,
    typename TIndex,
    typename TRange,
    typename TCategorie>
struct NextElement<
    boost::container::vector<TComponent>,
    TComponent,
    TIndex,
    TRange,
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<
        typename TContainerSize>
    HDINLINE
    TRange
    operator() (
        TContainer* container,
        TComponent*&,
        TIndex& idx, 
        TRange const & range,
        TContainerSize& size)
    {
        idx += range;
        return (idx >= size(container)) * (idx - (size(container)-1) );
    }
};

/**
 * @brief Implementation to check whether the end is reached. For further 
 * informations \see AfterLastElement.hpp
 */
template<
    typename TComponent,
    typename TIndex,
    typename TCategorie>
struct EndElement<
    boost::container::vector<TComponent>,
    TComponent,
    TIndex, 
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<typename TSizeFunction>
    HDINLINE
    bool
    test (
        TContainer* conPtr,
        TComponent* const &,
        TIndex const & idx,
        TSizeFunction const & size
    )
    const
    {
        return idx >= size(conPtr);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set(
        TContainer* conPtr,
        TComponent*&,
        TIndex & idx,
        TSizeFunction const & size
    )
    const
    {
        idx = size(conPtr);
    }
    
};

/**
 * @brief Implementation of the array like last element trait. For further details
 * \see LastElement.hpp
 */
template<
    typename TComponent,
    typename TIndex,
    typename TCategorie>
struct LastElement<
    boost::container::vector<TComponent>,
    TComponent,
    TIndex,
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (
        TContainer* conPtr,
        TComponent*&,
        TIndex& index,
        TSizeFunction& size)
    {
        index = size(conPtr) - 1;
    }
};

/**
 * @brief The implementation to get the last element in a array like data
 * structure. For futher details \see PreviousElement.hpp
 */
template<
    typename TComponent,
    typename TIndex,
    typename TRange,
    typename TCategorie>
struct PreviousElement<
    boost::container::vector<TComponent>,
    TComponent,
    TIndex,
    TRange,
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<typename T>
    HDINLINE
    int
    operator() (
        TContainer*,
        TComponent*&,
        TIndex& idx, 
        TRange const & jumpsize,
        T const &
    )
    {
        idx -= jumpsize;
        
        return (static_cast<int>(idx) < static_cast<int>(0)) * (-1 * static_cast<int>(idx));
    }
};

/**
 * @brief Implmentation to get check whether the iterator is on the element 
 * before the first one. \see REndElement.hpp
 */
template<
    typename TComponent,
    typename TIndex,
    typename TOffset,
    typename TCategorie>
struct REndElement<
    boost::container::vector<TComponent>,
    TComponent,
    TIndex,
    TOffset,
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<typename TSizeFunction>
    HDINLINE
    bool
    test (
        TContainer*,
        TComponent* const &,
        TIndex const & idx,
        TSizeFunction&
    )
    const
    {
        return static_cast<int>(idx) < static_cast<int>(0);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set (
        TContainer*,
        TComponent*&,
        TIndex & idx,
        TSizeFunction&
    )
    const
    {
        idx = static_cast<TIndex>(-1);
    }
};
}// namespace navigator
} // namespace traits
} // namespace deepiterator

/**
 * @brief Now we test the boost vector
 */


using namespace boost::unit_test;

typedef deepiterator::Particle<int_fast32_t, 2u> Particle;
typedef deepiterator::Frame<Particle, 10u> Frame;
typedef deepiterator::Supercell<Frame> Supercell;
typedef deepiterator::SupercellContainer<Supercell> SupercellContainer;

BOOST_AUTO_TEST_CASE(Frames)
{
    boost::container::vector<Frame> data(10);
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    
    auto && prescription = deepiterator::makeIteratorPrescription(
        deepiterator::makeAccessor(),
        deepiterator::makeNavigator(
            Offset(0),
            Jumpsize(1)),
        deepiterator::makeIteratorPrescription(
            deepiterator::makeAccessor(),
            deepiterator::makeNavigator(
                Offset(0),
                Jumpsize(1)),
            deepiterator::makeIteratorPrescription(
                deepiterator::makeAccessor(),
                deepiterator::makeNavigator(
                    Offset(0),
                    Jumpsize(1)))));
                                                           
    auto && view = deepiterator::makeView(data, prescription);
    uint sum = 0;
    for(auto && it = view.begin(); it != view.end(); ++it)
    {
        sum += *it;
    }
    
    uint sum_check = 0;
    for(auto i=0u; i<200u; ++i)
    {
        sum_check += i;
    }
    BOOST_TEST(sum == sum_check);
}
