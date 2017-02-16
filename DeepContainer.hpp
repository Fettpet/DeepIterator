#pragma once
#include "DeepIterator.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Collective.hpp"
namespace Data 
{
template<
    typename TContainer,
    typename TElement,
    Data::Direction Direction,
    typename Collectiv,
    size_t jumpSize
    >
struct DeepContainer;

// DeepIterator<Deepcontainer

/** **********************************************
 * @brief This iterator is the connector between two layers. 
 * @tparam TContainer The datatype of the input type: At the moment we have
 * Frame and Supercell Implemented.
 * @tparam TElement The return type. Implemented are particle frames and container
 * @tparam TDirection The direction of the iteration. 
 * @tparam jumpSize number of elements to jump over.
 * ********************************************/
template<
    typename TContainer,
    typename TElement,
    Data::Direction TDirection,
    typename TColl,
    size_t jumpSize>
struct DeepContainer
{
public:
    typedef TElement                                        ValueType; 
    typedef TContainer                                      InputType; 
    typedef InputType*                                      InputPointer;
    typedef TElement                                        ChildType; 
    typedef typename ChildType::InputType                   ChildInput;
    typedef Navigator<TContainer, TDirection, jumpSize>     NavigatorType;
    typedef Accessor<TContainer>                            AccessorType;
    
    typedef DeepIterator<InputType, AccessorType, NavigatorType, TColl, ChildType> iterator; 
    typedef DeepIterator<InputType, AccessorType, NavigatorType, TColl, ChildType> Iterator; 
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
    
    template<typename TOffset>
    iterator begin(const TOffset& offset)
    {
        return iterator(refContainer, offset);
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
    typename TCollective,
    unsigned nbParticleInFrame
    >
struct DeepContainer<
            Data::Frame<Particle<TPos, Dim>, nbParticleInFrame>, 
            Data::Particle<TPos, Dim>, 
            TDirection, 
            TCollective,
            jumpSize>
{
    typedef Particle<TPos, Dim>                                                     ValueType;
    typedef Frame<ValueType, nbParticleInFrame>                                     FrameType;
    typedef FrameType                                                               InputType;
    typedef Navigator<FrameType, TDirection, jumpSize>                              NavigatorType;
    typedef Accessor<FrameType>                                                     AccessorType;
    typedef DeepIterator<FrameType, AccessorType, NavigatorType, TCollective,Data::NoChild>     iterator;
    typedef DeepIterator<FrameType, AccessorType, NavigatorType, TCollective,Data::NoChild>     Iterator;
    /**
     * FrameType 
     */
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
    
    template<typename TOffset>
    iterator begin(const TOffset& offset) {
        return iterator(refContainer, offset);
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
    unsigned nbParticleInFrame,
    typename TCollective
    >
struct DeepContainer<
        Data::SuperCell<Data::Frame<TParticle, nbParticleInFrame> >,
        Data::Frame<TParticle, nbParticleInFrame>,
        TDirection,
        TCollective,
        jumpSize
        >
{
    typedef Frame<TParticle, nbParticleInFrame>                                         ValueType;
    typedef SuperCell<ValueType >                                                       ContainerType;
    typedef Navigator<ContainerType, TDirection, jumpSize>                              NavigatorType;
    typedef Accessor<ContainerType>                                                     AccessorType;
    typedef DeepIterator<ContainerType, AccessorType, NavigatorType, TCollective,  Data::NoChild>     iterator;
    typedef DeepIterator<ContainerType, AccessorType, NavigatorType, TCollective,  Data::NoChild>     Iterator;
    typedef DeepContainer<ContainerType, ValueType, TDirection, Data::Collectivity::NonCollectiv, jumpSize>               ThisType;
    typedef ContainerType                                                               InputType;
    
    DeepContainer(ContainerType& container):
        refContainer(container)
    {}
    
    iterator begin() {
        return iterator(refContainer.firstFrame);
    }
    
    template<typename TOffset>
    iterator begin(const TOffset& offset) {
        return iterator(refContainer.firstFrame, offset);
    }
    
    iterator end() {
        return iterator(nullptr);
    }
    
    ContainerType& refContainer;
    unsigned nbElem;
}; // DeepContainer

} // namespace Data