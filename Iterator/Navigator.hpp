#pragma once
#include "Policies.hpp"
#include "../PIC/Frame.hpp"

namespace Data 
{

/**
 * @brief The navigator is used to go to the next element
 * 
 */
template<typename TData,
         Data::Direction TDirection,
         size_t jumpSize>
struct Navigator;
    
/** ****************
 * @brief This Navigator can acess all Particles in a Frame
 *****************/
template<typename TParticle,
         size_t nbParticle,
         size_t jumpSize>
struct Navigator< Frame<TParticle, nbParticle>,Data::Forward, jumpSize>
{
public:
    template<typename TStore>
    void next(TStore& store) 
    {
        store.index += jumpSize;
    }
}; // Navigator<Forward, Frame, jumpSize>
    
template<typename TParticle,
         size_t nbParticle,
         size_t jumpSize>
struct Navigator< Frame<TParticle, nbParticle>,Data::Backward, jumpSize>
{
public:
    template<typename TStore>
    void next(TStore& store) 
    {
        store.index -= jumpSize;
    }
}; // Navigator<Backward, Frame, jumpSize>


/** ****************
 * @brief This Navigator can acess all Frames in a Supercell
 *****************/
template<typename TParticle,
         size_t nbParticle>
struct Navigator< Frame<TParticle, nbParticle>,Data::Forward, 1>
{
public:
    template<typename TStore>
    void next(TStore& store) 
    {
        store.ref = *(store.ref.nextFrame);
    }
}; // Navigator<Forward, Frame, jumpSize>
    
    
template<typename TParticle,
         size_t nbParticle>
struct Navigator< Frame<TParticle, nbParticle>,Data::Forward, 1>
{
public:
    template<typename TStore>
    void next(TStore& store) 
    {
        store.ref = *(store.ref.previousFrame);
    }
}; // Navigator<Forward, Frame, jumpSize>

}// namespace Data