/**
 * @author Sebastian Hahn t.hahn@hzdr.de
 * @brief The ComponentType trait gives information about the type of the 
 * components of a container. You need to implement a shape such that,
 * typedef ComponentType< ContainerType >::type YourComponentType;
 * is a valid and correct statement.
 * 
 */
#pragma once
#include "PIC/Frame.hpp"
#include "PIC/Supercell.hpp"
#include "PIC/Particle.hpp"
#include "PIC/SupercellContainer.hpp"

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

template<typename Supercell>
struct ComponentType< hzdr::SupercellContainer<Supercell> >
{
    typedef Supercell type;
};


}//traits
}//hzdr
