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
 * 5. NumberElements
 * <b> This is in each case needed </b>. 
 */
namespace hzdr 
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
struct NumberElements<boost::container::vector<T> >
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
 * 3. get
 * 4. equal
 * 5. ahead
 * 6. behind
 * Six for the navigator Behaviour
 * 7. firstElement
 * 8. nextElement
 * 9. afterLastElement
 * 10. lastElement 
 * 11. previousElement
 * 12. beforeFIrstElement
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
struct Get<
    boost::container::vector<TComponent> , 
    TComponent, 
    TIndex, 
    TCategorie
    >
{
    typedef boost::container::vector<TComponent> TContainer;
    
    HDINLINE
    TComponent&
    operator() (TContainer* con, TIndex const & idx)
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
        TIndex const & idx1, 
        TContainer * const con2, 
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
        TIndex const & idx1, 
        TContainer * const con2, 
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
        TIndex const & idx1, 
        TContainer * const con2, 
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
 * details \see FirstElement.hpp
 */
template<
    typename TComponent,
    typename TIndex,
    typename TCategorie>
struct FirstElement<
    boost::container::vector<TComponent>, 
    TIndex, 
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    HDINLINE
    void
    operator() (TContainer*, TIndex& idx)
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
struct AfterLastElement<
    boost::container::vector<TComponent>, 
    TIndex, 
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<typename TSizeFunction>
    HDINLINE
    bool
    test (TContainer* conPtr, TIndex const & idx, TSizeFunction const & size)
    const
    {
        return idx >= size(conPtr);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set(TContainer* conPtr, TIndex & idx, TSizeFunction const & size)
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
    TIndex,
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (TContainer* conPtr, 
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
    TIndex,
    TRange,
    TCategorie>
{
    typedef boost::container::vector<TComponent> TContainer;
    
    template<typename T>
    HDINLINE
    auto
    operator() (
        TContainer*, 
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
 * before the first one. \see BeforeFirstElement.hpp
 */
template<
    typename TComponent,
    typename TIndex,
    typename TOffset,
    typename TCategorie>
struct BeforeFirstElement<
    boost::container::vector<TComponent>, 
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
        TIndex const & idx,
        TSizeFunction&)
    const
    ->
    bool
    {
        return static_cast<int>(idx) < static_cast<int>(0);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set (TContainer*, TIndex & idx, TSizeFunction&)
    const
    {
        idx = static_cast<TIndex>(-1);
    }
};

    
}// namespace navigator

} // namespace traits

} // namespace hzdr


/**
 * now all traits are specified. We like to try it a little
 */

BOOST_AUTO_TEST_CASE(SimpleTest)
{
    typedef boost::container::vector<int> MyContainerType;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    
    
    auto && concept = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0),
            Jumpsize(1))
    );
    
    MyContainerType container{0,1,2,3,4,5,6,7,8,9,10};
    
    auto && view = hzdr::makeView(container, concept);
    int counter = 0;
    for(auto && it=view.begin(); it != view.end(); ++it)
    {
        BOOST_TEST( *it == counter++);
    }
    BOOST_TEST(counter == 11);
    
    for(auto && it=view.rbegin(); it != view.rend(); --it)
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
    typedef hzdr::Particle<ParticleProperty, 2u> Particle;
    typedef hzdr::Frame<Particle, 10u> Frame;
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
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    
    auto && concept = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0u),
            Jumpsize(1u)),
        hzdr::makeIteratorPrescription(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                Offset(0u),
                Jumpsize(1u)),
            hzdr::makeIteratorPrescription(
                hzdr::makeAccessor(),
                hzdr::makeNavigator(
                    Offset(0u),
                    Jumpsize(1u)))));
    
    auto && view = hzdr::makeView(container, concept);
    
    uint counter = 0u;
    for(auto && it = view.begin(); it != view.end(); ++it)
    {
        counter++;
        *it = 10 + counter;
    }
    BOOST_TEST(counter == (nbFrames - 1u) * 10u * 2u + particleLastFrame * 2u);
    
}
