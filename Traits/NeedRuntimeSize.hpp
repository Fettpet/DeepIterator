/**
 * @brief For iteratable datatypes the number of elements in the last element 
 * can be different. This size is known at runtime.
 * This trait decide wheter we can use the compiletime size or the
 * runtime size
 */ 

#pragma once
#include <PIC/Frame.hpp>
#include <PIC/Particle.hpp>
#include <PIC/Supercell.hpp>
#include <PIC/SupercellContainer.hpp>
#include "Definitions/hdinline.hpp"
namespace hzdr 
{

namespace traits
{

template<typename T>
struct NeedRuntimeSize;


template<typename Supercell>
struct NeedRuntimeSize<hzdr::SupercellContainer<Supercell> >
{
    static
    constexpr
    bool 
    test(...)
    {
        return true;
    }
};


template<typename TParticle, int_fast32_t nb>
struct NeedRuntimeSize<hzdr::Frame<TParticle, nb> >
{
    typedef hzdr::Frame<TParticle, nb>  Frame;
    typedef Frame*                      FramePtr;
    
    HDINLINE
    static
    bool 
    test(Frame const * const ptr)
    {
        return ptr->nextFrame == nullptr;
    }
}; // struct NeedRuntimeSize

template<typename TPosition, int_fast32_t dim>
struct NeedRuntimeSize<hzdr::Particle<TPosition, dim> >
{
    typedef hzdr::Particle<TPosition, dim>  Particle;
    typedef Particle*                          ParticlePtr;
    
     
    static
    constexpr
    bool 
    test(...)
    {
        return false;
    }
};





template<typename TFrame>
struct NeedRuntimeSize<hzdr::SuperCell<TFrame> >
{
    typedef hzdr::SuperCell<TFrame>            Supercell;
    typedef Supercell*                         SupercellPtr;
    
     
    static
    constexpr
    bool 
    test(Supercell const * const ptr)
    {
        return false;
    }
};

}//namespace traits
}//namespace hzdr
