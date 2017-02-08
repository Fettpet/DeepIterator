#pragma once
#include "../PIC/Frame.hpp"
#include "../PIC/Supercell.hpp"


namespace Data
{
template<typename TData>
struct Accessor;

template<typename TParticle, 
         size_t nbParticle>
struct Accessor<Frame<TParticle, nbParticle> >
{
    
    
    template<TStorage>
    TParticle
    inline
    get(TStorage& store)
    {
        return store.ref[store.index];
    }
       
}; // Accessor < Frame >

template<typename TFrame>
struct Accessor<SuperCell<TFrame> >
{
    
    
    template<TStorage>
    TFrame
    inline
    get(TStorage& store)
    {
        return store.ref;
    }
       
}; // Accessor < Frame >

}// namespace Data