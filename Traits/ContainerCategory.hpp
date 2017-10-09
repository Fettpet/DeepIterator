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
#include "Iterator/Categorie/ArrayLike.hpp"
#include "Iterator/Categorie/DoublyLinkListLike.hpp"

namespace hzdr 
{
namespace traits
{

    
template<typename T>
struct ContainerCategory
{
    typedef T type;
};
/*
template<typename TParticle, int_fast32_t nb>
struct ContainerCategory<hzdr::Frame<TParticle, nb> >
{
    typedef hzdr::container::categorie::ArrayLike type;
};

template<typename TPos, int_fast32_t dim>
struct ContainerCategory<hzdr::Particle<TPos, dim> >
{
    typedef hzdr::container::categorie::ArrayLike type;
};
    
template<typename TSuperCell>
struct ContainerCategory<hzdr::SupercellContainer<TSuperCell> >
{
    typedef hzdr::container::categorie::ArrayLike type;
};
  */
}// namespace traits
    
}// namespace hzdr
