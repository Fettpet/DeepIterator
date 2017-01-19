#pragma once
#include "DeepIterator.hpp"
#include "Frame.hpp"
#include "Particle.hpp"
template<
    typename TContainer,
    typename TElement>
struct DeepContainer;


template<
    typename TContainer,
    typename TElement>
struct DeepContainer
{
public:
    typedef DeepIterator<TElement> iterator;
    
public:
    DeepContainer(TContainer& container):
        refContainer(container)
    {}
    
    iterator begin() {
        return DeepIterator<TElement>(*(refContainer.begin()));
    }
    
    
    iterator end() {
        return DeepIterator<TElement>(refContainer.size());
    }
    
    
    
protected:
    TContainer& refContainer;
};


/** ****************************************************************************
 *@brief specialisation for Particle in frames
 ******************************************************************************/

template<
    typename TPos,
    unsigned Dim,
    unsigned nbParticleInFrame
    >
struct DeepContainer<Frame<Particle<TPos, Dim>, nbParticleInFrame>, Particle<TPos, Dim> >
{
    typedef Particle<TPos, Dim> TElement;
    typedef Frame<Particle<TPos, Dim>, nbParticleInFrame> TContainer;
    typedef DeepIterator<TElement> iterator;
    
    
    DeepContainer(TContainer& container, unsigned nbElem):
        refContainer(container), nbElem(nbElem)
    {}
    
    iterator begin() {
        return DeepIterator<TElement>((refContainer.particles[0]));
    }
    
    
    iterator end() {
        if(refContainer.nextFrame != nullptr)
        {
            return DeepIterator<TElement>(nbParticleInFrame);
        }
        else
        {
            return DeepIterator<TElement>(nbElem);
        }
    }
    
    TContainer& refContainer;
    unsigned nbElem;
};

/** ****************************************************************************
 *@brief specialisation for Frames in Suprecell
 ******************************************************************************/

template<
    typename TParticle,
   
    unsigned nbParticleInFrame
    >
struct DeepContainer<SuperCell<Frame<TParticle, nbParticleInFrame> >, Frame<TParticle, nbParticleInFrame> >
{
    typedef Frame<TParticle, nbParticleInFrame>             TElement;
    typedef SuperCell<Frame<TParticle, nbParticleInFrame> > TContainer;
    typedef DeepIterator<TContainer>                        iterator;
    
    
    DeepContainer(TContainer& container):
        refContainer(container)
    {}
    
    iterator begin() {
        return DeepIterator<TContainer>(refContainer.firstFrame);
    }
    
    
    iterator end() {
        return DeepIterator<TContainer>(nullptr);
    }
    
    TContainer& refContainer;
    unsigned nbElem;
};