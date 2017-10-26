#include <map>
/**
 * @brief We use this test to give you an advice how you can create an own 
 * categorie. We like to use the map out of the std library as a categorie. The 
 * namespacing is like the namespacing in the iterator package. The first thing 
 * you need to do, give the categorie a name. 
 */

namespace hzdr 
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
 * We use the std iterator to iterate over the data struc
 */

template<
    typename Key, 
    typename T>
struct IndexType<std::map<key, T> > 
{
    typedef std::map<key,T>::iterator type; 
};

template<
    typename T>
struct RangeType<boost::container::vector<T>, void> 
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
    int
    operator() (
        TContainer*, 
        TIndex& idx, 
        TRange const & offset,
        TRange const & jumpsize,
        T const &
               )
    {
        idx -= jumpsize;
        
        return (static_cast<int>(idx) < static_cast<int>(offset)) * (static_cast<int>(offset) - static_cast<int>(idx));
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
    bool
    test (TContainer*, TIndex const & idx, TOffset const & offset, TSizeFunction&)
    const
    {
        return static_cast<int>(idx) < static_cast<int>(offset);
    }
    
    template<typename TSizeFunction>
    HDINLINE
    void
    set (TContainer*, TIndex & idx, TOffset const &, TSizeFunction&)
    const
    {
        idx = static_cast<TIndex>(-1);
    }
};

} // namespace hzdr
