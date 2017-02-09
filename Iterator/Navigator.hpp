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
         unsigned jumpSize>
struct Navigator;
    
/** ****************
 * @brief This Navigator can acess all Particles in a Frame
 *****************/
template<typename TParticle,
         unsigned nbParticle,
         unsigned jumpSize>
struct Navigator< Data::Frame<TParticle, nbParticle>,Data::Direction::Forward, jumpSize>
{
public:
    template<typename TIndex>
    static
    void 
    inline
    next(TIndex& index) 
    {
        index += jumpSize;
    }
    
    static
    unsigned 
    inline 
    first(const unsigned& offset, const unsigned& nbParticleInFrame)
    {
        return offset;
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
template<typename TParticle,
         unsigned nbParticle,
         unsigned jumpSize>
struct Navigator< Data::Frame<TParticle, nbParticle>,Data::Direction::Backward, jumpSize>
{
public:
    template<typename TIndex>
    static
    void 
    inline
    next(TIndex& index) 
    {
        index -= jumpSize;
    }
    
    
    static
    unsigned 
    inline 
    first(const unsigned& offset, const unsigned& nbParticleInFrame)
    {
        return nbParticleInFrame-offset;
    }
    
    
}; // Navigator<Backward, Frame, jumpSize>


/** ****************
 * @brief This Navigator can acess all Frames in a Supercell
 *****************/
template<typename TFrame>
struct Navigator< Data::SuperCell<TFrame>, Data::Direction::Forward, 1>
{
    typedef Data::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
    
public:
    static
    void 
    inline
    next(FramePointer& ptr) 
    {
        ptr = ptr->nextFrame;
    }
    
    static 
    FramePointer
    inline
    first(const SuperCellType* supercell)
    {
        return supercell->firstFrame;
    }
    
    static 
    FramePointer
    inline
    first(nullptr_t)
    {
        return nullptr;
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
    
template<typename TFrame>
struct Navigator< Data::SuperCell<TFrame>, Data::Direction::Backward, 1>
{
    typedef Data::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
public:
    

    static
    void
    inline
    next(FramePointer& ptr) 
    {
        ptr = ptr->previousFrame;
    }
    
    static 
    FramePointer
    
    first(const SuperCellType* supercell)
    {
        if(supercell != nullptr)
        {
            return supercell->lastFrame;
        }
        return nullptr;
    }
    
    static 
    FramePointer
    
    first(nullptr_t)
    {
        return nullptr;
    }
}; // Navigator<Forward, Frame, jumpSize>

}// namespace Data