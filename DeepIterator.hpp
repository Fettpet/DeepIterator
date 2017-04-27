/**
 * @author Sebastian Hahn (t.hahn@hzdr.de )
 * @brief The DeepIterator class is used to iterator over interleaved data 
 * structures. The simplest example is for an interleaved data structure is 
 * std::vector< std::vector< int > >. The deepiterator iterates over all ints 
 * within the structure.
 * The iterator support lists and index based access. Both are specialised. 
 * Because the implementation of interleaved and flat is different, we need four
 * implemtations of the iterator:
 * 1. flat and list based
 * 2. flat and index based
 * 3. interleaved and list based
 * 4. interleaved and index based
 *
 * The iterator use the trait \b IsIndexable to decide wether the datastructure 
 * is array like or list like. 
 * This implementation is special for the datastructurs of PIConGPU. 
 * 
 */
#include <sstream>
#pragma once
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "PIC/Supercell.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Collective.hpp"
#include "Traits/NeedRuntimeSize.hpp"
#include <boost/iterator/iterator_concepts.hpp>
#include "Iterator/Wrapper.hpp"
#include "Traits/IsIndexable.hpp"
#include <limits>
#include <cassert>
#include <type_traits>
#include "Traits/NumberElements.hpp"


namespace hzdr 
{
/**
 * @tparam TElement is the type of the element 
 * @tparam TChild ist ein virtueller Container oder NoChild
 */

template<typename TElement, 
         typename TAccessor, 
         typename TNavigator, 
         typename TCollective, 
         typename TRuntimeVariables,
         typename TChild,
         typename TEnable = void>
struct DeepIterator;




/** ************************************+
 * @brief The flat implementation with indexable element type
 * ************************************/
template<typename TElement,
        typename TAccessor, 
        typename TNavigator,
        typename TRuntimeVariables,
        typename TCollective>
struct DeepIterator<TElement, 
                    TAccessor, 
                    TNavigator, 
                    TCollective, 
                    TRuntimeVariables,
                    hzdr::NoChild,
                    typename std::enable_if<hzdr::traits::IsIndexable<TElement>::value>::type >
{
// datatypes
public:
    typedef TElement                                ElementType;
    typedef typename std::remove_reference<typename TElement::ValueType>::type            ValueType;
    typedef ValueType*                              ValuePointer;
    typedef ValueType&                              ValueReference;
    typedef ValueType                               ReturnType;
    typedef TAccessor                               Accessor;
    typedef TNavigator                              Navigator;
    typedef TCollective                             Collecter;
    typedef Wrapper< ValueType, TCollective>        WrapperType;
    typedef traits::NeedRuntimeSize<ElementType>    RuntimeSize;

 
// functios
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */
    DeepIterator(nullptr_t, 
                 const int_fast32_t& nbElems
    ):
        index(nbElems),
        nbElemsInLast(0)
    {}

    DeepIterator():
        index(0),
        nbElemsInLast(0)
        {}
    
    DeepIterator(ElementType* _ptr, 
                 const int_fast32_t& nbElemsInLast, 
                 const TRuntimeVariables& runtimeVariables
                ):
                 index(Navigator::first(runtimeVariables)),
                 nbElemsInLast(nbElemsInLast),
                 runtimeVariables(runtimeVariables)
                 
    {
  //      assert(not (std::is_same<TCollective, Collectivity::None>::value and runtimeVariables.getJumpsize() == 1));
        if(coll.isMover())
        {
            ptr = _ptr;
        }
        coll.sync();
    }
    
    /**
     * @brief goto the next element
     */

    DeepIterator&
    operator++()
    {
        coll.sync();
        if(coll.isMover())
        {
            Navigator::next(ptr, index, runtimeVariables);

        }
        coll.sync();
        return *this;
    }
    
   
    WrapperType
    operator*()
    {
        if(RuntimeSize::test(ptr))
        {
            return WrapperType(Accessor::get(ptr, index, runtimeVariables.getNbElements()));
        }
        else 
        {
            
            auto elem =  traits::NumberElements<ElementType>::value;
            return WrapperType(Accessor::get(ptr, index, elem));
        }
    }
    
    
    bool
    operator!=(const DeepIterator& other)
    const
    {
// #pragma omp critical
//         std::cout << std::boolalpha << "first test " << (index < other.index + runtimeVariables.getJumpsize() - index % runtimeVariables.getJumpsize()) << " 2. test" << (index > -1 * runtimeVariables.getJumpsize()) << " index " << index <<std::endl;  
        // ceil index such that all threads work
        
        return index < other.index + runtimeVariables.getJumpsize() - other.index % runtimeVariables.getJumpsize()
             and index >= -1 * (runtimeVariables.getJumpsize() - other.index % runtimeVariables.getJumpsize());
    }

        
    bool
    operator!=(nullptr_t)
    const
    {
        return true;
    }
    
    DeepIterator& operator=(const DeepIterator& other)
    {
        ptr = other.ptr;
        index = other.index;
        //nbElems = other.nbElems;
        
        return *this;
    }
    
    
    void setPtr(ValuePointer inPtr)
    {
        ptr = inPtr;
    }
    
protected:
    Collecter coll;
    ElementType* ptr;
    int index;
    const int_fast32_t nbElemsInLast;
    TRuntimeVariables runtimeVariables;
    
private:

}; // struct DeepIterator



/** ************************************+
 * @brief The flat implementation with list like type
 * ************************************/
template<typename TElement,
        typename TAccessor, 
        typename TNavigator,
        typename TRuntimeVariables,
        typename TCollective>
struct DeepIterator<TElement, 
                    TAccessor, 
                    TNavigator, 
                    TCollective, 
                    TRuntimeVariables,
                    hzdr::NoChild,
                    typename std::enable_if<not hzdr::traits::IsIndexable<TElement>::value, void>::type >
{
// datatypes
public:
    typedef TElement                            ElementType;
    typedef typename TElement::ValueType        ValueType;
    typedef ValueType*                          ValuePointer;
    typedef ValueType&                          ValueReference;
    typedef ValueType                           ReturnType;
    typedef TAccessor                           Accessor;
    typedef TNavigator                          Navigator;
    typedef TCollective                         Collecter;
    typedef Wrapper< ValueType, TCollective>    WrapperType;
// functions 
    static_assert(std::is_same<typename TAccessor::ReturnType, ValueType>::value, "Returntype of accessor must be the same as Valuetype of TElement");
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */
    DeepIterator(nullptr_t, 
                 const int_fast32_t& nbElems
    ):
    waitAtEnd(false)
    {}

    
   DeepIterator(ElementType* _ptr, 
                 const int_fast32_t& nbElemsInLast, 
                 const TRuntimeVariables& runtimeVariables
                ):
                 
                 runtimeVariables(runtimeVariables),
                 waitAtEnd(false)
                 
    {
        if(coll.isMover())
        {
            ptr = Navigator::first(_ptr, runtimeVariables);
        }
        coll.sync();
    }
    
    /**
     * @brief goto the next element
     */

    DeepIterator&
    operator++()
    {

        coll.sync();
        if(coll.isMover())
        {
            waitAtEnd = Navigator::next(ptr, runtimeVariables);
        }
        coll.sync();

        return *this;
    }
    
   
    WrapperType
    operator*()
    {
        return WrapperType(Accessor::get(ptr));
    }
    
    
    bool
    operator!=(const DeepIterator& other)
    const
    {
#pragma omp critical
            std::cout << "waitAtEnd " << waitAtEnd << " id " << omp_get_thread_num() << std::endl;
        return ptr != nullptr and not waitAtEnd;
    }

        
    bool
    operator!=(nullptr_t)
    const
    {
        return true;
    }
    
    void setPtr(ValuePointer inPtr)
    {
        ptr = inPtr;
    }
    
protected:
    Collecter coll;
    ValueType* ptr = nullptr;
    TRuntimeVariables runtimeVariables;
    bool waitAtEnd;
private:
}; // struct DeepIterator







/** ************************************+
 * @brief The nested implementation with indexable element type
 * ************************************/
template<typename TElement,
        typename TAccessor, 
        typename TNavigator,
        typename TCollective,
        typename TRuntimeVariables,
        typename TChild>
struct DeepIterator<TElement, 
                    TAccessor, 
                    TNavigator, 
                    TCollective, 
                    TRuntimeVariables,
                    TChild,
                    typename std::enable_if<hzdr::traits::IsIndexable<TElement>::value>::type >
{
// datatypes
public:
    typedef TElement                                ElementType;
    typedef typename TElement::ValueType            ValueType;

    typedef ValueType*                              ValuePointer;
    typedef ValueType&                              ValueReference;
    typedef TAccessor                               Accessor;
    typedef TNavigator                              Navigator;
    typedef TCollective                             Collecter;

    typedef traits::NeedRuntimeSize<ElementType>    RuntimeSize;
// child things
    typedef TChild                                  ChildView;
    typedef typename TChild::Iterator               ChildIterator;
    typedef typename ChildIterator::ReturnType      ReturnType;
    typedef Wrapper< ReturnType, TCollective>       WrapperType;
// functios
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */
    DeepIterator(nullptr_t, 
                 const int_fast32_t& nbElems
    ):
        index(nbElems)

    {}

    DeepIterator():
        index(0)
        {}
    
    
    DeepIterator(ElementType* _ptr, 
                 const int_fast32_t& nbElemsInLast,
                 const TRuntimeVariables& runtimeVariables,
                 ChildView view):
                 index(Navigator::first(runtimeVariables)),
                 runtimeVariables(runtimeVariables),
                 childView(view),
                 childIter(view.begin())
                 
    {
        if(coll.isMover())
        {
            ptr = _ptr;
        }
        coll.sync();
        if(traits::NeedRuntimeSize<ElementType>::test(ptr))
        {
            childView = ChildView(Accessor::get(ptr, index, runtimeVariables.getNbElements()));

        }
        else 
        {
            auto elem =  traits::NumberElements<ElementType>::value;
            childView = ChildView(Accessor::get(ptr, index, elem));
        }
        childIter = childView.begin();
    }
    
    /**
     * @brief goto the next element
     */

    DeepIterator&
    operator++()
    {
        coll.sync();
        if(coll.isMover())
        {
             ++childIter;
            if(not (childIter != childView.end()))
            {
                Navigator::next(ptr, index, runtimeVariables);
                if(traits::NeedRuntimeSize<ElementType>::test(ptr))
                {
                    childView = ChildView(Accessor::get(ptr, index, runtimeVariables.getNbElements()));
                }
                else 
                {
                    auto elem =  traits::NumberElements<ElementType>::value;
                    childView = ChildView(Accessor::get(ptr, index, elem));
                    
                }
             //   childView = ChildView(Accessor::get(ptr, index));
                childIter = childView.begin();
            }
                
        }
        coll.sync();
        return *this;
    }
    
   
    WrapperType
    operator*()
    {
      //  auto t = Accessor::get(ptr, index);
        return *childIter;
    }
    
    
    bool
    operator!=(const DeepIterator& other)
    const
    {
        std::cout << "index " << index << " other.index " << other.index << std::endl;
    //    if(ptr == nullptr) return false;
        return index < other.index + runtimeVariables.getJumpsize() - 1;
        
    }

        
    bool
    operator!=(nullptr_t)
    const
    {
        return true;
    }
    
protected:
    
    Collecter coll;
    ElementType* ptr = nullptr;
    int_fast32_t index;
    TRuntimeVariables runtimeVariables;
    ChildView childView;
    ChildIterator childIter;
    
private:

}; // struct DeepIterator




/** ************************************+
 * @brief The nested implementation with list like element type
 * ************************************/
template<typename TElement,
        typename TAccessor, 
        typename TNavigator,
        typename TCollective,
        typename TRuntimeVariables,
        typename TChild>
struct DeepIterator<TElement, 
                    TAccessor, 
                    TNavigator, 
                    TCollective, 
                    TRuntimeVariables,
                    TChild,
                    typename std::enable_if<not hzdr::traits::IsIndexable<TElement>::value>::type >
{
// datatypes
    
public:
    typedef TElement                                ElementType;
    typedef typename std::remove_reference<typename TElement::ValueType>::type            ValueType;

    typedef ValueType*                              ValuePointer;
    typedef ValueType&                              ValueReference;
    typedef TAccessor                               Accessor;
    typedef TNavigator                              Navigator;
    typedef TCollective                             Collecter;

    typedef traits::NeedRuntimeSize<ElementType>    RuntimeSize;
// child things
    typedef TChild                                  ChildView;
    typedef typename TChild::Iterator               ChildIterator;
    typedef typename ChildIterator::ReturnType      ReturnType;
    typedef typename ChildIterator::WrapperType     WrapperType;

    // tests
    static_assert(std::is_same<typename TAccessor::ReturnType, ValueType>::value, "Returntype of accessor must be the same as Valuetype of TElement");
    
    // functions
    
public:

/**
 * @brief creates an virtual iterator. This one is used to specify a last element
 * @param nbElems number of elements within the datastructure
 */
    DeepIterator(nullptr_t, 
                 const int_fast32_t& nbElems
    )

    {}

    DeepIterator()
        {}
    
    DeepIterator(ElementType* _ptr, 
                 const int_fast32_t& nbElems,
                 const int_fast32_t& nbElemsInLast)
    {
        if(coll.isMover())
        {
            ptr = Navigator::first(_ptr);
        }
        coll.sync();
        
    }
    
    
    DeepIterator(ElementType* ptr2, 
                 const int_fast32_t& nbElems,
                 TRuntimeVariables runtimeVariables,
                 ChildView view):
                 
                 childView(view),
                 childIter(view.begin()),
                 runtimeVariables(runtimeVariables)
                
                 
    {
        if(coll.isMover())
        {
            ptr = Navigator::first(ptr2, runtimeVariables);
        }
        coll.sync();
        childView = ChildView(ptr);
        childIter = childView.begin();
    }
    
    /**
     * @brief goto the next element
     */

    DeepIterator&
    operator++()
    {
        coll.sync();
        if(coll.isMover())
        {

             ++childIter;
            if(not (childIter != childView.end()))
            {

                Navigator::next(ptr,runtimeVariables);
                
                childView = ChildView(Accessor::get(ptr));
                childIter = childView.begin();
            }
                
        }
        coll.sync();
        return *this;
    }
    
   
    WrapperType
    operator*()
    {
      //  auto t = Accessor::get(ptr, index);
        return *childIter;
    }
    
    
    bool
    operator!=(const DeepIterator& other)
    const
    {
        
        return ptr != nullptr;
    }

        
    bool
    operator!=(nullptr_t)
    const
    {
        return true;
    }
    
protected:
    Collecter coll;
    ValueType* ptr;
    ChildView childView;
    ChildIterator childIter;
    TRuntimeVariables runtimeVariables;
    
private:

}; // struct DeepIterator

}// namespace hzdr