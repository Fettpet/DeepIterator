#pragma once
#include "DeepIterator.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"


namespace Data 
{
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
    typedef TElement                                        value_type; // DeepContainer
    typedef TContainer                                      input_type; // Supercell
    typedef TElement                                        child_type; // Deepcontainer
    typedef DeepContainer<TContainer, 
                          typename child_type::input_type>  iter_type;
    
    typedef DeepIterator<iter_type>                         iterator; // DeepIterator<Deepcontainer

    
    typedef typename TElement::container_type   container_type;
    
    
public:
    DeepContainer(input_type& container):
        refContainer(container)
    {}
    
    iterator begin() {
        auto test = (iter_type(refContainer));
        return iterator(test);
    }
    
    
    iterator end() {
        return iterator(refContainer.end());
    }
    
    
    
    
protected:
    input_type& refContainer;
}; // DeepContainer


/** ****************************************************************************
 *@brief specialisation for Particle in frames
 ******************************************************************************/

template<
    typename TPos,
    unsigned Dim,
    unsigned nbParticleInFrame
    >
struct DeepContainer<Data::Frame<Particle<TPos, Dim>, nbParticleInFrame>, Data::Particle<TPos, Dim> >
{
    typedef Particle<TPos, Dim>                         value_type;
    typedef Frame<value_type, nbParticleInFrame>        container_type;
    typedef DeepIterator<value_type>                    iterator;
    typedef DeepContainer<container_type, value_type>   this_type;
    typedef container_type                              input_type;
    
    
    DeepContainer(container_type& container, unsigned nbElem):
        refContainer(container), nbElem(nbElem)
    {}
    
    DeepContainer(const DeepContainer & other):
        refContainer(other.refContainer),
        nbElem(other.nbElem)
    {}
    
    iterator begin() {
        return DeepIterator<value_type>(refContainer.particles[0]);
    }
    
    
    iterator end() {
        if(refContainer.nextFrame != nullptr)
        {
            return DeepIterator<value_type>(nbParticleInFrame);
        }
        else
        {
            return DeepIterator<value_type>(nbElem);
        }
    }
    
    container_type& refContainer;
    unsigned nbElem;
}; // DeepContainer

/** ****************************************************************************
 *@brief specialisation for Frames in Suprecell
 ******************************************************************************/

template<
    typename TParticle,
   
    unsigned nbParticleInFrame
    >
struct DeepContainer<Data::SuperCell<Data::Frame<TParticle, nbParticleInFrame> >, Data::Frame<TParticle, nbParticleInFrame> >
{
    typedef Frame<TParticle, nbParticleInFrame>         value_type;
    typedef SuperCell<value_type >                      container_type;
    typedef DeepIterator<container_type>                iterator;
    typedef DeepContainer<container_type, value_type>   this_type;
    typedef container_type                              input_type;
    DeepContainer(container_type& container):
        refContainer(container)
    {}
    
    iterator begin() {
        return DeepIterator<container_type>(refContainer.firstFrame);
    }
    
    
    iterator end() {
        return DeepIterator<container_type>(nullptr);
    }
    
    container_type& refContainer;
    unsigned nbElem;
}; // DeepContainer

} // namespace Data