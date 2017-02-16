#pragma once
#include "Policies.hpp"
#include "PIC/Frame.hpp"

namespace hzdr 
{

/**
 * @brief The navigator is used to go to the next element
 * 
 */
template<typename Thzdr,
         hzdr::Direction TDirection,
         unsigned jumpSize>
struct Navigator;
    
/** ****************
 * @brief This Navigator can acess all Particles in a Frame
 *****************/
template<typename TParticle,
         unsigned nbParticle,
         unsigned jumpSize>
struct Navigator< hzdr::Frame<TParticle, nbParticle>,hzdr::Direction::Forward, jumpSize>
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
struct Navigator< hzdr::Frame<TParticle, nbParticle>,hzdr::Direction::Backward, jumpSize>
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
template<typename TFrame,
         unsigned jumpSize>
struct Navigator< hzdr::SuperCell<TFrame>, hzdr::Direction::Forward, jumpSize>
{
    typedef hzdr::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
    
public:
    static
    void 
    inline
    next(FramePointer& ptr) 
    {
        for(size_t i=0; i<jumpSize; ++i)
        {
            if(ptr == nullptr) continue;
            ptr = ptr->nextFrame;
        }
        
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
    
    
template<typename TFrame, unsigned jumpSize>
struct Navigator< hzdr::SuperCell<TFrame>, hzdr::Direction::Backward, jumpSize>
{
    typedef hzdr::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
public:
    

    static
    void
    inline
    next(FramePointer& ptr) 
    {
        for(size_t i=0; i<jumpSize; ++i)
        {
            if(ptr == nullptr) continue;    
            ptr = ptr->previousFrame;
        }
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

}// namespace hzdr