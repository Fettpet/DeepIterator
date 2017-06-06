#pragma once
#include "PIC/Frame.hpp"
#include "PIC/Supercell.hpp"
#include "PIC/Particle.hpp"

namespace hzdr 
{
namespace traits 
{
template<typename T>
struct ComponentType;

template<typename Particle, int_fast32_t size>
struct ComponentType< hzdr::Frame<Particle, size> >
{
    typedef Particle type;
};

template<typename Frame>
struct ComponentType< hzdr::SuperCell<Frame> >
{
    typedef Frame type;
};

template<typename TElem, int_fast32_t size>
struct ComponentType< hzdr::Particle< TElem, size> >
{
    typedef TElem type;
};


}//traits
}//hzdr
