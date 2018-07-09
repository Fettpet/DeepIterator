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

 /**
 * @brief Within this test we show how to use your own container type. We show 
 * this with the boost-vector-container. See this test as a tutorial.
 */
#define BOOST_TEST_MODULE OwnContainer
// test stuff
#include <boost/test/included/unit_test.hpp>
// The container class
#include <boost/container/vector.hpp>
// include all the needed trait headers
#include "deepiterator/traits/Traits.hpp"



#include "deepiterator/DeepIterator.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/PIC/Frame.hpp"
/**
 * Each new Container need to specify the following traits:
 * 1. Componenttype
 * 2. HasConstantSize
 * 3. IsBidirectional
 * 4. IsRandomAccessable
 * 5. Size
 * <b> This is in each case needed </b>. 
 */
namespace deepiterator 
{
namespace traits 
{
template<
    typename T, 
    typename SFIANE>
struct IsBidirectional<boost::container::vector<T>, SFIANE >
{
    static const bool value = true;
};

template<
    typename T,
    typename SFIANE>
struct IsRandomAccessable<boost::container::vector<T>, SFIANE >
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
struct ComponentType<boost::container::vector<T> >
{
    typedef T type;
};

template<typename T>
struct Size<boost::container::vector<T> >
{
    typedef boost::container::vector<T> Container;
    
    HDINLINE
    int_fast32_t
    operator()( Container* element)
    const
    {
        return element->size();
    }
};


/**
 * After these four traits are specified, you can set the containerCategory, if
 * your xontainer has the same properties like the categorie. If this is the 
 * case, you are finished. Our boost vector hasnt a own categorie. So you have 
 * to specify the following traits:
 * 1. IndexType,
 * 2. RangeType,
 * Four for the accessor Behaviour
 * 3. At
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
 */

template<
    typename T,
    typename SFIANE
>
struct IndexType<boost::container::vector<T>, SFIANE> 
{
    typedef int_fast32_t type; 
};

template<
    typename T,
    typename SFIANE
>
struct RangeType<
    boost::container::vector<T>, 
    SFIANE
> 
{
    typedef int_fast32_t type; 
};

namespace accessor 
{
/**
 * @brief Get the value out of the container at the current iterator position.
 * \see Get.hpp
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
        TComponent* &,
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
        TComponent* &,
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
 * informations \see EndElement.hpp
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
        TComponent* &,
        TIndex& index,
        TSizeFunction& size
    )
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
    TCategorie
>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<typename T>
    HDINLINE
    auto
    operator() (
        TContainer*,
        TComponent* &,
        TIndex& idx, 
        TRange const & jumpsize,
        T const &
    )
    const
    ->
    int
    {
        idx -= jumpsize;
        
        return (static_cast<int>(idx) < 0) * ( -1 * static_cast<int>(idx));
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
    auto
    test (
        TContainer*,
        TComponent* const &,
        TIndex const & idx,
        TSizeFunction&
    )
    const
    ->
    bool
    {
        return static_cast<int>(idx) < static_cast<int>(0);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set (
        TContainer*,
        TComponent* &,
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
 * now all traits are specified. We like to try it a little
 */

BOOST_AUTO_TEST_CASE(SimpleTest)
{
    typedef boost::container::vector<int> MyContainerType;
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    
    
    
    auto && concept = deepiterator::makeIteratorPrescription(
        deepiterator::makeAccessor(),
        deepiterator::makeNavigator(
            Offset(0),
            Jumpsize(1))
    );
    
    MyContainerType container{0,1,2,3,4,5,6,7,8,9,10};
    
    auto && view = deepiterator::makeView(container, concept);
    int counter = 0;
    for(auto && it=view.begin(); it != view.end(); ++it)
    {
        BOOST_TEST( *it == counter++);
    }
    BOOST_TEST(counter == 11);
    
    for(auto && it=view.rbegin(); it != view.rend(); ++it)
    {
        BOOST_TEST( *it == --counter);
    }
    BOOST_TEST(counter == 0);
}


/**
 * Now we can use boost vector to store frames and go over all particles
 */
BOOST_AUTO_TEST_CASE(ComplexTest)
{
    // define the needed datatypes
    typedef int_fast32_t ParticleProperty;
    typedef deepiterator::Particle<ParticleProperty, 2u> Particle;
    typedef deepiterator::Frame<Particle, 10u> Frame;
    typedef boost::container::vector<Frame> MyContainerType;
    
    // set the test data
    const uint_fast32_t nbFrames = 5u;
    const uint_fast32_t particleLastFrame = 6u;
    MyContainerType container;
    for(uint i=0u; i<nbFrames-1; ++i)
    {
        container.push_back(Frame());
    }
    container.push_back(Frame(particleLastFrame));
    
    // define offset and jumpsize types
    typedef deepiterator::SelfValue<uint_fast32_t> Offset;
    typedef deepiterator::SelfValue<uint_fast32_t> Jumpsize;
    
    auto && concept = deepiterator::makeIteratorPrescription(
        deepiterator::makeAccessor(),
        deepiterator::makeNavigator(
            Offset(0u),
            Jumpsize(1u)),
        deepiterator::makeIteratorPrescription(
            deepiterator::makeAccessor(),
            deepiterator::makeNavigator(
                Offset(0u),
                Jumpsize(1u)),
            deepiterator::makeIteratorPrescription(
                deepiterator::makeAccessor(),
                deepiterator::makeNavigator(
                    Offset(0u),
                    Jumpsize(1u)))));
    
    auto && view = deepiterator::makeView(container, concept);
    
    uint counter = 0u;
    for(auto && it = view.begin(); it != view.end(); ++it)
    {
        counter++;
        *it = 10 + counter;
    }
    BOOST_TEST(counter == (nbFrames - 1u) * 10u * 2u + particleLastFrame * 2u);
    
}
