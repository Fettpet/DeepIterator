#pragma once 
#include "../PIC/Frame.hpp"

namespace Data 
{
    
    template<typename TData>
    struct Storage;
    
    /**
     * 
     */
    template<typename TParticle, 
            size_t nbParticle>
    struct Storage<Frame<TParticle, nbParticle> >
    {
        // Data structures
        typedef Frame<TParticle, nbParticle>    StoredType;
        typedef StoredType*                     StoredPointer;
        typedef StoredType&                     StoredReference;
        typedef TParticle                       ParticleType;
        
        // Constructors
        Storage() = default;
        
        Storage(StoredReference store, size_t index):
            index(index),
            ref(store)
        {}
        
        
        // Variables
        size_t index;
        StoredReference ref;
    };// Storage < Frame >
    
    template<typename TFrame>
    struct Storage<SuperCell<TFrame> >
    {
        // Data structures
        typedef TFrame                          StoredType;
        typedef StoredType*                     StoredPointer;
        typedef StoredType&                     StoredReference;

        // Constructor
        Storage() = default;
        Storage(StoredReference store):
            ref(store)
        {}
        
        // Variables
        StoredReference ref;
    }; // Storage < SuperCell >
}