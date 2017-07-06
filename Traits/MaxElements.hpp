/**
 * \class MaxElements
 * @author Sebastian Hahn t.hahn@hzdr.de
 * @brief This is a helper class to get the maximum number of elements within
 * a container. This helper class has one function, size(const containertype&), 
 * which determine the size of the container. If it is not possible, for example
 * an linked list, it return std::limits::<uint>::max()
 */


#pragma once
#include <PIC/Frame.hpp>
#include <PIC/Particle.hpp>
#include <PIC/Supercell.hpp>
#include <PIC/SupercellContainer.hpp>
#include "Definitions/hdinline.hpp"
#include <iomanip>
#include <limits>

namespace hzdr
{
template<typename T>
struct MaxElements;
    
    
template<typename Supercell>
struct MaxElements<hzdr::SupercellContainer<Supercell> >
{
    typedef hzdr::SupercellContainer<Supercell> SupercellContainer;
    
    
    uint_fast32_t
    size(const SupercellContainer element&)
    const
    {
        return element.getNbSupercells();
    }
};

template<typename Frame>
struct MaxElements<hzdr::SuperCell<Frame> >
{
    typedef hzdr::SuperCell<Frame> Supercell;
    
    HDINLINE
    uint_fast32_t
    constexpr
    size(const Supercell element&)
    {
        return std::numeric_limits<uint_fast32_t>::max();
    } 
}


template<typename TParticle, int_fast32_t nb>
struct MaxElements<hzdr::Frame<TParticle, nb> >
{
    typedef hzdr::Frame<TParticle, nb>  Frame;
    typedef Frame*                      FramePtr;
    
    HDINLINE
    constexpr
    uint_fast32_t
    size(const Frame&)
    {
        return nb;
    }
}; 

template<typename TPosition, int_fast32_t dim>
struct MaxElements<hzdr::Particle<TPosition, dim> >
{
    typedef hzdr::Particle<TPosition, dim>  Particle;
    typedef Particle*                       ParticlePtr;
    
     
    HDINLINE
    constexpr
    uint_fast32_t
    size(const Particle&)
    {
        return dim;
    }
};




}
