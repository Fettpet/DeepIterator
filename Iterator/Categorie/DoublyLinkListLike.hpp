#pragma once
#include "PIC/Supercell.hpp"

namespace hzdr
{
namespace container 
{
namespace categorie
{

struct DoublyLinkListLike;



template<typename TContainer>
struct NextElement;

template<typename TFrame>
struct NextElement<hzdr::SuperCell<TFrame> >
{
    typedef TFrame          ComponentType;
    typedef ComponentType*  ComponentPtr;
    typedef hzdr::SuperCell<TFrame> ContainerType;
    typedef ContainerType* ContainerPtr;
    
    ComponentPtr 
    operator() (ComponentPtr comPtr)
    {
        return comPtr->nextFrame;
    }
};

template<typename TContainer>
struct PreviousElement;


template<typename TFrame>
struct PreviousElement<hzdr::SuperCell<TFrame> >
{
    typedef TFrame          ComponentType;
    typedef ComponentType*  ComponentPtr;
    typedef hzdr::SuperCell<TFrame> ContainerType;
    typedef ContainerType* ContainerPtr;
    
    ComponentPtr 
    operator() (ComponentPtr comPtr)
    {
        return comPtr->previousFrame;
    }
};


template<typename TContainer>
struct FirstElement;

template<typename TFrame>
struct FirstElement<hzdr::SuperCell<TFrame> >
{
    typedef TFrame          ComponentType;
    typedef ComponentType*  ComponentPtr;
    typedef hzdr::SuperCell<TFrame> ContainerType;
    typedef ContainerType* ContainerPtr;
    
    ComponentPtr 
    operator() (ContainerPtr conPtr)
    {
        return conPtr->firstFrame;
    }
};

template<typename TContainer>
struct LastElement;

template<typename TFrame>
struct LastElement<hzdr::SuperCell<TFrame> >
{
    typedef TFrame          ComponentType;
    typedef ComponentType*  ComponentPtr;
    typedef hzdr::SuperCell<TFrame> ContainerType;
    typedef ContainerType* ContainerPtr;
    
    ComponentPtr 
    operator() (ContainerPtr conPtr)
    {
        return conPtr->lastFrame;
    }
};

template<typename>
struct EndElementReached;

template<typename TFrame>
struct EndElementReached<hzdr::SuperCell<TFrame> >
{
    typedef TFrame          ComponentType;
    typedef ComponentType*  ComponentPtr;
    typedef hzdr::SuperCell<TFrame> ContainerType;
    typedef ContainerType* ContainerPtr;
    
    bool 
    operator() (ComponentPtr comPtr)
    {
        return comPtr == nullptr;
    }
};

template<typename >
struct BeforeElementReached;


template<typename TFrame>
struct BeforeElementReached<hzdr::SuperCell<TFrame> >
{
    typedef TFrame          ComponentType;
    typedef ComponentType*  ComponentPtr;
    typedef hzdr::SuperCell<TFrame> ContainerType;
    typedef ContainerType* ContainerPtr;
    
    bool 
    operator() (ComponentPtr comPtr)
    {
        return comPtr == nullptr;
    }
};

} // namespace categorie

} // namespace contaienr
}// namespace hzdr
