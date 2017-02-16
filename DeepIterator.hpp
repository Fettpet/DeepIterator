#pragma once
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "PIC/Supercell.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Collective.hpp"
#include <boost/iterator/iterator_concepts.hpp>





namespace Data 
{
/**
 * @tparam TChild ist ein virtueller Container oder NoChild
 */

template<typename TElement,
        typename TAccessor, 
        typename TNavigator, 
        typename TCollective,
        typename TChild>
struct DeepIterator;


/** ************************************+
 * @brief specialication for Particle in Frame
 * ************************************/
template<typename TParticle,
        typename TAccessor, 
        typename TNavigator,
        typename TCollective,
        unsigned nbParticles>
struct DeepIterator<Data::Frame<TParticle, nbParticles>, TAccessor, TNavigator, TCollective, Data::NoChild>
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
    typedef TCollective                     Collecter;
 
// functios
public:
    DeepIterator(const size_t index):
        ptr(nullptr), index(index)
    {}
    DeepIterator(const FrameReference ref, const size_t index):
        ptr(&ref), index(index)
    {}
    
    DeepIterator(const DeepIterator& other) = default;
    
    DeepIterator& operator=(const DeepIterator& other)
    {
        if(coll.isMover())
        {
            coll = other.coll;
            ptr = other.ptr;
            index = other.index;    
        }
        coll.sync();
        return *this;
        
    }
    /**
     * @brief goto the next element
     */

    DeepIterator&
    operator++()
    {
        if(coll.isMover())
        {
            Navigator::next(index);
        }
        coll.sync();
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
    Collecter coll;
    FramePointer ptr;
    size_t index;
    
}; // struct DeepIterator


/** ****************************************************************************
 * @brief 
 * ****************************************************************************/

template<typename TFrame,
        typename TNavigator,
        typename TCollective,
        typename TAccessor>
struct DeepIterator<Data::SuperCell<TFrame>, TAccessor, TNavigator, TCollective, NoChild>
{
public:
    typedef TFrame                          ValueType;
    typedef ValueType*                      ValuePointer;
    typedef ValueType&                      ValueReference;
    typedef TAccessor                       Accessor;
    typedef TNavigator                      Navigator;
    typedef TCollective                     Collecter;
    
public:
    
    DeepIterator(ValuePointer ptr):
        ptr(ptr)
    {}
    
    template<typename TOffset>
    DeepIterator(ValuePointer ptr, const TOffset& offset):
        ptr(ptr)
    {
        for(TOffset i=static_cast<TOffset>(0); i < offset; ++i)
        {
            if(coll.isMover())
                Navigator::next(ptr);
        }
        coll.sync();

    }
    
    DeepIterator(ValueType& value):
        ptr(&value)
    {}
    
    template<typename TOffset>
    DeepIterator(ValueType& value, const TOffset& offset):
        ptr(&value)
    {
        for(TOffset i=static_cast<TOffset>(0); i < offset; ++i)
        {
            if(coll.isMover())
                Navigator::next(ptr);
        }
        coll.sync();
    }
    
    
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
        if(coll.isMover())
        {
                Navigator::next(ptr);
        }
        coll.sync();
        
        return *this;
    }
    
    
protected:
    Collecter coll;
    ValuePointer ptr;
};// DeepIterator



template<typename TFrame,
         typename TNavigator,
         typename TAccessor,
         typename TCollective,
         typename TChild>
struct DeepIterator<Data::SuperCell<TFrame>, TAccessor, TNavigator, TCollective, TChild>
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
    typedef DeepIterator<Data::SuperCell<TFrame>, TAccessor, TNavigator, TCollective, TChild> ThisType;
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