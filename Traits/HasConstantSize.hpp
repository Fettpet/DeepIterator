/**
 * \struct HasConstantSize
 * @author Sebastian Hahn (t.hahn < at > hzdr) 
 * @brief This trait decide whether a container has a constant size, or not. A 
 * size of a container is constant, if number of elements within the container
 * is known as compile time.
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
struct HasConstantSize
{
    const static bool value = false;
};

/*

template<>
struct HasConstantSize<details::UndefinedType >
{
    const static bool value = false;
};

template<typename TParticle, int_fast32_t nb>
struct HasConstantSize<hzdr::Frame<TParticle, nb> >
{
    const static bool value = true;
};

template<typename TPos, int_fast32_t dim>
struct HasConstantSize<hzdr::Particle<TPos, dim> >
{
    const static bool value = true;
};
    
template<typename TFrame>
struct HasConstantSize<hzdr::Supercell<TFrame> >
{
    const static bool value = false;
};
    
template<typename TSupercell>
struct HasConstantSize<hzdr::SupercellContainer<TSupercell> >
{
    const static bool value = false;
};
  */
}// namespace traits
    
}// namespace hzdr

