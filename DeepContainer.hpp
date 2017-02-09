#pragma once
#include "DeepIterator.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include "Iterator/Policies.hpp"

namespace Data 
{
template<
    typename TContainer,
    typename TElement,
    Data::Direction Direction,
    size_t jumpSize
    >
struct DeepContainer;


template<
    typename TContainer,
    typename TElement,
    Data::Direction TDirection,
    size_t jumpSize>
struct DeepContainer
{
public:
    typedef TElement                                        ValueType; // DeepContainer
    typedef TContainer                                      InputType; // Supercell
    typedef InputType*                                      InputPointer;
    typedef TElement                                        ChildType; // Deepcontainer
    typedef typename ChildType::InputType                   ChildInput;
    typedef Navigator<TContainer, TDirection, jumpSize>     NavigatorType;
    typedef Accessor<TContainer>                            AccessorType;
    
    typedef DeepIterator<InputType, AccessorType, NavigatorType, ChildType> iterator; // DeepIterator<Deepcontainer

    /*
    typedef typename TElement::container_type   container_type;
    */
    
public:
    DeepContainer(InputType& container):
        refContainer(&container)
    {}
    
    DeepContainer(InputPointer con):
        refContainer(con)
        {}
    
    
    
    iterator begin() {
  //      auto test = (ValueType(refContainer));
        return iterator(refContainer);
    }
    
    
    iterator end() {
        return iterator(nullptr);
    }
    
    
    
    
protected:
    InputPointer refContainer;
}; // DeepContainer


/** ****************************************************************************
 *@brief specialisation for Particle in frames
 ******************************************************************************/

template<
    typename TPos,
    Data::Direction TDirection,
    size_t jumpSize,
    unsigned Dim,
    unsigned nbParticleInFrame
    >
struct DeepContainer<Data::Frame<Particle<TPos, Dim>, nbParticleInFrame>, Data::Particle<TPos, Dim>, TDirection, jumpSize>
{
    typedef Particle<TPos, Dim>                                                     ValueType;
    typedef Frame<ValueType, nbParticleInFrame>                                     FrameType;
    typedef FrameType                                                               InputType;
    typedef Navigator<FrameType, TDirection, jumpSize>                              NavigatorType;
    typedef Accessor<FrameType>                                                     AccessorType;
    typedef DeepIterator<FrameType, AccessorType, NavigatorType, Data::NoChild>     iterator;
    
    
    DeepContainer(FrameType& container, unsigned nbElem):
        refContainer(container), nbElem(nbElem)
    {}
    
    DeepContainer(const DeepContainer & other):
        refContainer(other.refContainer),
        nbElem(other.nbElem)
    {}
    
    iterator begin() {
        return iterator(refContainer, 0);
    }
    
    
    iterator end() {
        if(refContainer.nextFrame != nullptr)
        {
            return iterator(nbParticleInFrame);
        }
        else
        {
            return iterator(nbElem);
        }
    }
    
    FrameType& refContainer;
    unsigned nbElem;
}; // DeepContainer

/** ****************************************************************************
 *@brief specialisation for Frames in Suprecell
 ******************************************************************************/

template<
    typename TParticle,
    Data::Direction TDirection,
    size_t jumpSize,
    unsigned nbParticleInFrame
    >
struct DeepContainer<
        Data::SuperCell<Data::Frame<TParticle, nbParticleInFrame> >,
        Data::Frame<TParticle, nbParticleInFrame>,
        TDirection,
        jumpSize
        >
{
    typedef Frame<TParticle, nbParticleInFrame>                                         ValueType;
    typedef SuperCell<ValueType >                                                       ContainerType;
    typedef Navigator<ContainerType, TDirection, jumpSize>                              NavigatorType;
    typedef Accessor<ContainerType>                                                     AccessorType;
    typedef DeepIterator<ContainerType, AccessorType, NavigatorType, Data::NoChild>     iterator;
    typedef DeepContainer<ContainerType, ValueType, TDirection, jumpSize>               ThisType;
    typedef ContainerType                                                               InputType;
    
    DeepContainer(ContainerType& container):
        refContainer(container)
    {}
    
    iterator begin() {
        return iterator(refContainer.firstFrame);
    }
    
    
    iterator end() {
        return iterator(nullptr);
    }
    
    ContainerType& refContainer;
    unsigned nbElem;
}; // DeepContainer

} // namespace Data