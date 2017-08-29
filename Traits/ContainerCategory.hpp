/**
 * \struct IsIndexable
 * @author Sebastian Hahn (t.hahn < at > hzdr) 
 * @brief The function of this trait is to decide wheter a container is random 
 * accessable. A container is random accessable if it has the operator[] overloaded.
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
namespace details
{
struct ArrayBased;
struct ListBased;
}

template<typename T>
struct ContainerCategory;



  
template<typename TParticle, int_fast32_t nb>
struct ContainerCategory<hzdr::Frame<TParticle, nb> >
{
    typedef traits::details::ArrayBased type;
};

template<typename TPos, int_fast32_t dim>
struct ContainerCategory<hzdr::Particle<TPos, dim> >
{
    typedef traits::details::ArrayBased type;
};
    
template<typename TFrame>
struct ContainerCategory<hzdr::SuperCell<TFrame> >
{
    typedef traits::details::ListBased type;

};
    
template<typename TSuperCell>
struct ContainerCategory<hzdr::SupercellContainer<TSuperCell> >
{
    typedef traits::details::ArrayBased type;
};
  
}// namespace traits
    
}// namespace hzdr
