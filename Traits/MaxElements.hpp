/**
 * \class MaxElements
 * @author Sebastian Hahn t.hahn@hzdr.de
 * @brief This is a helper class to get the maximum number of elements within
 * a container. This helper class has one function, size(const containertype&), 
 * which determine the size of the container. If it is not possible, for example
 * an linked list, it set the value to RuntimeSize
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
namespace traits
{
template<typename T>
struct MaxElements;
const int_fast32_t RuntimeSize = -1;
    
template<typename Supercell>
struct MaxElements<hzdr::SupercellContainer<Supercell> >
{
    static const int_fast32_t value = RuntimeSize;
};

template<typename Frame>
struct MaxElements<hzdr::Supercell<Frame> >
{
    
    static const int_fast32_t value = RuntimeSize;
};


template<typename TParticle, int_fast32_t nb>
struct MaxElements<hzdr::Frame<TParticle, nb> >
{
    static const int_fast32_t value = nb;
}; 

template<typename TPosition, int_fast32_t dim>
struct MaxElements<hzdr::Particle<TPosition, dim> >
{
    static const int_fast32_t value = dim;
};


} // namespace traits

} // namespace hzdr
