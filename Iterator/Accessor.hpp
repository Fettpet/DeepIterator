#pragma once
#include "../PIC/Frame.hpp"
#include "../PIC/Supercell.hpp"
#include <iostream>

namespace Data
{
template<typename TData>
struct Accessor;



template<typename TParticle, 
         unsigned nbParticle>
struct Accessor<Data::Frame<TParticle, nbParticle> >
{
     

    typedef TParticle                       ParticleType;
    typedef Data::Frame<TParticle, nbParticle>    FrameType;
    typedef FrameType*                      FramePointer;
    typedef TParticle                       ReturnType;
    typedef ReturnType&                     ReturnReference;
   
    
    
    template<typename TIndex>
    static
    ReturnType&
    
    get(FramePointer frame, const TIndex& index)
    {
        return (*frame)[index];
    }
    
       
}; // Accessor < Frame >

template<typename TFrame>
struct Accessor<SuperCell<TFrame> >
{
    typedef TFrame                          FrameType;
    typedef FrameType*                      FramePointer;
    typedef FrameType                       ReturnType;
    typedef ReturnType&                     ReturnReference;
    
    
    Accessor() = default;
    static
    ReturnReference
    inline
    get(FramePointer frame)
    {
        return *frame;
    }
    
    
    static
    ReturnReference
    inline
    get(FrameType& frame)
    {
        return frame;
    }
    
    static
    const
    ReturnReference
    inline
    get(const FrameType& frame)
    {
        return frame;
    }
    
       
}; // Accessor < Frame >

}// namespace Data