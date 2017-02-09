#pragma once
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "PIC/Supercell.hpp"
#include "Iterator/Policies.hpp"
#include <boost/iterator/iterator_concepts.hpp>





namespace Data 
{
/**
 * @tparam TChild ist ein virtueller Container oder NoChild
 */

template<typename TElement,
        typename TAccessor, 
        typename TNavigator, 
        typename TChild>
struct DeepIterator;


/** ************************************+
 * @brief specialication for Particle in Frame
 * ************************************/
template<typename TParticle,
        typename TAccessor, 
        typename TNavigator,
        unsigned nbParticles>
struct DeepIterator<Data::Frame<TParticle, nbParticles>, TAccessor, TNavigator, Data::NoChild>
{
// datatypes
public:
    typedef Frame<TParticle, nbParticles>   FrameType;
    typedef FrameType*                      FramePointer;
    typedef FrameType&                      FrameReference;
    typedef TParticle                       ValueType;
    typedef ValueType*                      ValuePointer;
    typedef ValueType&                      ValueReference;
    typedef TAccessor                       Accessor;
    typedef TNavigator                      Navigator;
 
// functios
public:
    DeepIterator(const size_t index):
        ptr(nullptr), index(index)
    {}
    DeepIterator(const FrameReference ref, const size_t index):
        ptr(&ref), index(index)
    {}
    
    DeepIterator(const DeepIterator& other) = default;
    
    DeepIterator& operator=(const DeepIterator&) = default;
    /**
     * @brief goto the next element
     */

    DeepIterator&
    operator++()
    {
        Navigator::next(index);
        return *this;
    }
    
   
    ValueReference
    operator*()
    {
        return Accessor::get(ptr, index);
    }
    
    const
    ValueReference
    operator*()
    const
    {
        return Accessor::get(ptr, index);
    }
    
    bool
    operator!=(const DeepIterator& other)
    const
    {
        return index < other.index;
    }

        
    bool
    operator!=(nullptr_t)
    const
    {
        return false;
    }
    
protected:
    FramePointer ptr;
    size_t index;
    
}; // struct DeepIterator


/** ****************************************************************************
 * @brief 
 * ****************************************************************************/

template<typename TFrame,
        typename TNavigator,
        typename TAccessor>
struct DeepIterator<Data::SuperCell<TFrame>, TAccessor, TNavigator, NoChild>
{
public:
    typedef TFrame                          ValueType;
    typedef ValueType*                      ValuePointer;
    typedef ValueType&                      ValueReference;
    typedef TAccessor                       Accessor;
    typedef TNavigator                      Navigator;
    
public:
    
    DeepIterator(ValuePointer ptr):
        ptr(ptr)
    {}
    
    DeepIterator(ValueType& value):
        ptr(&value)
    {}
    
    const 
    ValueReference
    operator*()
    const
    {
        return Accessor::get(ptr);
    }
    
    ValueReference
    operator*()
    {
        return Accessor::get(ptr);
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
        return false;
    }
    
    DeepIterator&
    operator++()
    {
        Navigator::next(ptr);
        return *this;
    }
    
    
protected:
    ValuePointer ptr;
};// DeepIterator



template<typename TFrame,
         typename TNavigator,
         typename TAccessor,
         typename TChild>
struct DeepIterator<Data::SuperCell<TFrame>, TAccessor, TNavigator, TChild>
{
public:
    typedef Data::SuperCell<TFrame>             SuperCellType;
    typedef SuperCellType                       InputType;
    typedef TFrame                              ValueType;
    typedef ValueType*                          ValuePointer;
    typedef ValueType&                          ValueReference;
    typedef TAccessor                           Accessor;
    typedef TNavigator                          Navigator;
    typedef TChild                              ChildContainer;
    typedef typename TChild::iterator           ChildIterator;
    typedef typename ChildContainer::ValueType  ResultType;     
    typedef DeepIterator<Data::SuperCell<TFrame>, TAccessor, TNavigator, TChild> ThisType;
public:
    
    DeepIterator(nullptr_t t):
        nbElem(0),
        ptr(nullptr)
    {}
    
    DeepIterator(InputType* in):
        nbElem(in->nbParticlesInLastFrame),
        ptr(Navigator::first(in))
    {
        std::cout << "Supercell with Child" << std::endl;
    }
    
    TChild
    operator*()
    {
        return TChild(Accessor::get(ptr), nbElem);
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
        return false;
    }
    
    ThisType&
    operator++()
    {
        Navigator::next(ptr);
        return *this;
    }
    
    
protected:
    unsigned nbElem;
    ValuePointer ptr;
};// DeepIterator < Supercell , Child >
    
    
    
} // namespace Data