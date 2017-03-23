    #pragma once
#include <PIC/Frame.hpp>
#include <PIC/Particle.hpp>
#include <PIC/SupercellContainer.hpp>

namespace hzdr 
{
namespace traits
{
template<typename T>
struct NumberElements
{
    static const uint_fast32_t value = 0;
}; // NumberElements
  
template<typename TParticle, uint_fast32_t nb>
struct NumberElements<hzdr::Frame<TParticle, nb> >
{
    static const uint_fast32_t value = nb;
}; // struct NumberElements

template<typename TPos, uint_fast32_t nb>
struct NumberElements<hzdr::Particle<TPos, nb> >
{
    static const uint_fast32_t value = nb;
}; // struct NumberElements

    
} // namespace traits

}// namespace hzdr
