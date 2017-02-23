/**
 * @author Sebastian Hahn (t.hahn@hzdr.de )
 * @brief The DeepView provides functionality for the DeepIterator. The first 
 * one is the construction of the DeepIterator type. This includes the navigator
 * and the accessor. The second part of the functionality is providing the begin
 * and end functions.
 * The import template arguments are TContainer and TElement. 
 * 
 */

#pragma once
#include "DeepIterator.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Collective.hpp"
namespace hzdr 
{
    
    
    
template<
    typename TContainer,
    typename TElement,
    hzdr::Direction Direction,
    typename Collectiv,
    size_t jumpSize
    >
struct DeepView;


/** **********************************************
 * @brief This view is the connector between two layers. 
 * @tparam TContainer The datatype of the input type: At the moment we have
 * Frame and Supercell Implemented.
 * @tparam TElement The return type. Implemented are particle frames and container
 * @tparam TDirection The direction of the iteration. 
 * @tparam jumpSize number of elements to jump over.
 * ********************************************/
template<
    typename TContainer,
    typename TElement,
    hzdr::Direction TDirection,
    typename TColl,
    size_t jumpSize>
struct DeepView
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
    DeepView(InputType& container):
        refContainer(&container)
    {}
    
    DeepView(InputPointer con):
        refContainer(con)
        {}
    
    DeepView& operator=(const DeepView&) = default;
    
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
}; // 


/** ****************************************************************************
 *@brief specialisation for Particle in frames
 ******************************************************************************/

template<
    typename TPos,
    hzdr::Direction TDirection,
    size_t jumpSize,
    unsigned Dim,
    typename TCollective,
    unsigned nbParticleInFrame
    >
struct DeepView<
            hzdr::Frame<Particle<TPos, Dim>, nbParticleInFrame>, 
            hzdr::Particle<TPos, Dim>, 
            TDirection, 
            TCollective,
            jumpSize>
{
    typedef Particle<TPos, Dim>                                                                 ValueType;
    typedef ValueType                                                                           ReturnType;
    typedef Frame<ValueType, nbParticleInFrame>                                                 FrameType;
    typedef FrameType                                                                           InputType;
    typedef Navigator<FrameType, TDirection, jumpSize>                                          NavigatorType;
    typedef Accessor<FrameType>                                                                 AccessorType;
    typedef DeepIterator<FrameType, AccessorType, NavigatorType, TCollective,hzdr::NoChild>     iterator;
    typedef DeepIterator<FrameType, AccessorType, NavigatorType, TCollective,hzdr::NoChild>     Iterator;
    /**
     * FrameType 
     */
    DeepView(FrameType& container, unsigned nbElem):
        refContainer(&container), nbElem(nbElem)
    {}
    
    DeepView(const  DeepView& other):
        refContainer(other.refContainer),
        nbElem(other.nbElem)
    {}
    
    DeepView(nullptr_t, unsigned):
        refContainer(nullptr), 
        nbElem(0)
        {}
    
    DeepView& operator=(const DeepView&) = default;
    
    iterator begin() {
        return iterator(*refContainer, 0);
    }
    
    template<typename TOffset>
    iterator begin(const TOffset& offset) {
        return iterator(*refContainer, offset);
    }
    
    
    iterator end() {
        if(refContainer->nextFrame != nullptr)
        {
            return iterator(nbParticleInFrame);
        }
        else
        {
            return iterator(nbElem);
        }
    }
    
    FrameType* refContainer;
    unsigned nbElem;
}; // 

/** ****************************************************************************
 *@brief specialisation for Frames in Suprecell
 ******************************************************************************/

template<
    typename TParticle,
    hzdr::Direction TDirection,
    size_t jumpSize,
    unsigned nbParticleInFrame,
    typename TCollective
    >
struct DeepView<
        hzdr::SuperCell<hzdr::Frame<TParticle, nbParticleInFrame> >,
        hzdr::Frame<TParticle, nbParticleInFrame>,
        TDirection,
        TCollective,
        jumpSize
        >
{
    typedef Frame<TParticle, nbParticleInFrame>                                                         ValueType;
    typedef SuperCell<ValueType >                                                                       ContainerType;
    typedef Navigator<ContainerType, TDirection, jumpSize>                                              NavigatorType;
    typedef Accessor<ContainerType>                                                                     AccessorType;
    typedef DeepIterator<ContainerType, AccessorType, NavigatorType, TCollective,  hzdr::NoChild>       iterator;
    typedef DeepIterator<ContainerType, AccessorType, NavigatorType, TCollective,  hzdr::NoChild>       Iterator;
    typedef DeepView<ContainerType, ValueType, TDirection, hzdr::Collectivity::NonCollectiv, jumpSize>  ThisType;
    typedef ContainerType                                                                               InputType;
    
    DeepView():
        refContainer(nullptr),
        nbElem(0)
    {}
    
    DeepView(ContainerType& container):
        refContainer(&container)
    {}
    
    DeepView(ContainerType* container):
        refContainer(container)
    {}
    
    DeepView& operator=(const DeepView&) = default;
    
    iterator begin() {
        return iterator(refContainer->firstFrame);
    }
    
    template<typename TOffset>
    iterator begin(const TOffset& offset) {
        return iterator(refContainer->firstFrame, offset);
    }
    
    iterator end() {
        return iterator(nullptr);
    }
    
    ContainerType* refContainer;
    unsigned nbElem;
}; // 

} // namespace hzdr