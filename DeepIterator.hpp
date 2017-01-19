#pragma once
#include "Frame.hpp"
#include "Supercell.hpp"
template<
    typename TElement>
struct DeepIterator;

/**
 * @brief Der DeepIterator ist ein Iterator über den DeepContainer. 
 * 
 */
template<
    typename TElement>
struct DeepIterator
{
public:
    typedef TElement        value_type;
    typedef TElement&       reference;
    typedef const reference const_reference;
    typedef TElement*       pointer;
    
public:
    
    DeepIterator(const size_t pos):
        pos(pos)
    {}
    
    DeepIterator(value_type& value):
        pos(0), ptr(&value)
    {}
    
    DeepIterator(value_type&& value):
        pos(0), ptr(&value)
    {}
    
    const_reference
    operator*()
    const
    {
        return ptr[pos];
    }
    
    reference
    operator*()
    {
        return ptr[pos];
    }
    
    bool
    operator!=(const DeepIterator& other)
    {
        return pos != other.pos;
    }
    
    void
    operator++()
    {
        pos++;
    }
    
    
protected:
    size_t pos;
    TElement* ptr;
};


/** ***************************************************************************
 * @brief Deep Iterator Ausprägung für Frames 
 ****************************************************************************/

template<
    typename TParticle,
    unsigned NbParticleInFrame>
struct DeepIterator<Frame<TParticle, NbParticleInFrame> >
{
public:
    typedef TParticle        value_type;
    typedef TParticle&       reference;
    typedef const reference const_reference;
    typedef TParticle*       pointer;
    
public:
    
    DeepIterator(const size_t pos):
        pos(pos)
    {}
    
    DeepIterator(value_type& value):
        pos(0), ptr(&value)
    {}
    
    DeepIterator(value_type&& value):
        pos(0), ptr(&value)
    {}
    
    const_reference
    operator*()
    const
    {
        return ptr[pos];
    }
    
    reference
    operator*()
    {
        return ptr[pos];
    }
    
    bool
    operator!=(const DeepIterator& other)
    {
        return pos != other.pos;
    }
    
    void
    operator++()
    {
        pos++;
    }
    
    
protected:
    size_t pos;
    TParticle* ptr;
};

/** ****************************************************************************
 * Ausprägung für Frames in Superzellen
 * ****************************************************************************/

template<
    typename TFrame>
struct DeepIterator<SuperCell<TFrame> >
{
public:
    typedef TFrame          value_type;
    typedef TFrame&         reference;
    typedef const reference const_reference;
    typedef TFrame*         pointer;
    
public:
    
    DeepIterator(pointer ptr):
        ptr(ptr)
    {}
    
    DeepIterator(value_type& value):
        ptr(&value)
    {}
    
    DeepIterator(value_type&& value):
        ptr(&value)
    {}
    
    const_reference
    operator*()
    const
    {
        return *ptr;
    }
    
    reference
    operator*()
    {
        return *ptr;
    }
    
    bool
    operator!=(const DeepIterator& other)
    {
        return ptr != nullptr;
    }
    
    void
    operator++()
    {
        ptr = ptr->nextFrame;
    }
    
    
protected:
    pointer ptr;
};