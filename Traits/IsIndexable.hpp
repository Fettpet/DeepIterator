/**
 * @author Sebastian Hahn (t.hahn<at>hzdr) 
 * @brief The function of this trait is to decide wheter a container is random 
 * accessable. A container is random accessable if it has the operator[] overloaded
 * 
 */

#pragma once
#include <PIC/Frame.hpp>
#include <PIC/Particle.hpp>
#include <PIC/Supercell.hpp>
#include <PIC/SupercellContainer.hpp>
namespace hzdr 
{
namespace traits
{
    template<typename T>
    struct IsIndexable;
  
    template<typename TParticle, uint_fast32_t nb>
    struct IsIndexable<hzdr::Frame<TParticle, nb> >
    {
        static const bool value = true;
    };

    template<typename TPos, uint_fast32_t dim>
    struct IsIndexable<hzdr::Particle<TPos, dim> >
    {
        static const bool value = true;
    };
    
    template<typename TFrame>
    struct IsIndexable<hzdr::SuperCell<TFrame> >
    {
        static const bool value = false;
    };
    
    template<typename TSuperCell>
    struct IsIndexable<hzdr::SupercellContainer<TSuperCell> >
    {
        static const bool value = true;
    };
  
}// namespace traits
    
}// namespace hzdr