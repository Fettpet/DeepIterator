/**
 * @author Sebastian Hahn
 * @brief A PIConGPU like datastructure. The supercell contains some frames.
 * The frames are in a linked list. Each frame has the pointer nextFrame and 
 * previousFrame. Only the lastFrame frame is not full with particles. The supercell
 * stores the number of particles in the lastFrame frame. Each supercell has two 
 * pointers to frame: firstFrame and lastFrame.
 */
#pragma once
#include "Definitions/hdinline.hpp"

namespace hzdr
{
template<typename TFrame>
struct SuperCell
{

    typedef TFrame frame_type;
    typedef TFrame FrameType;
    typedef TFrame ValueType;
    
    HDINLINE 
    SuperCell():
        firstFrame(nullptr),
        lastFrame(nullptr)
    {}
    
    HDINLINE 
    SuperCell(const SuperCell & other)
    {
        firstFrame = other.firstFrame;
        lastFrame = other.lastFrame;
    }
    
    HDINLINE 
    SuperCell( SuperCell && other)
    {
        firstFrame = other.firstFrame;
        lastFrame = other.lastFrame;
        other.firstFrame = nullptr;
        other.lastFrame = nullptr;
    }
    
    HDINLINE
    ~SuperCell() 
    {
        TFrame* cur = firstFrame;
        while(cur != nullptr)
        {
            TFrame* buffer = cur->nextFrame;
            delete cur;
            cur = buffer;
        }
    }
    
    HDINLINE
    SuperCell& 
    operator=(const SuperCell& other)
    {
        firstFrame = other.firstFrame;
        lastFrame = other.lastFrame;
        return *this;
    }
    
        
    HDINLINE
    SuperCell& 
    operator=( SuperCell&& other)
    {
        
        firstFrame = other.firstFrame;
        lastFrame = other.lastFrame;
        other.firstFrame = nullptr;
        other.lastFrame = nullptr;
        return *this;
    }
    
    /**
     * @param nbFrames: number of frames within the supercell,
     * @param nbParticle number of particles in the lastFrame frame
     */
    HDINLINE
    SuperCell(uint32_t nbFrames, uint32_t nbParticles):
        firstFrame(new TFrame())
    {
        TFrame *curFrame;
        curFrame = firstFrame;
        for(uint32_t i=1; i<nbFrames; ++i)
        {
            curFrame->nextFrame = new TFrame();
            curFrame->nextFrame->previousFrame = curFrame;
            curFrame = curFrame->nextFrame;
        }
        curFrame->nbParticlesInFrame = nbParticles;
        lastFrame = curFrame;
        
        for(uint32_t i=nbParticles; i<TFrame::maxParticlesInFrame; ++i)
        {
            for(uint32_t dim=0; dim < TFrame::Dim; ++dim)
                lastFrame->particles[i].data[dim] = -1;
        }
        
    }
    
    TFrame *firstFrame = nullptr;
    TFrame *lastFrame = nullptr;
 //   uint32_t nbParticlesInLastFrame;
}; // struct SuperCell

// traits
namespace traits 
{
template<typename>
struct IsBidirectional;
    
template<
    typename TFrame>
struct IsBidirectional<SuperCell<TFrame> >
{
    static const bool value = true;
};


template<typename>
struct IsRandomAccessable;

template<
    typename TFrame>
struct IsRandomAccessable<SuperCell<TFrame> >
{
    static const bool value = true;
};
    
template<typename>
struct HasConstantSize;

template<
    typename TFrame>
struct HasConstantSize<SuperCell<TFrame> >
{
    static const bool value = true;
};

template<typename>
struct ComponentType;

template<
    typename TFrame>
struct ComponentType<SuperCell<TFrame> >
{
    typedef TFrame type;
};

} // namespace traits


template<typename TFrame>
HDINLINE
std::ostream& operator<<(std::ostream& out, const SuperCell<TFrame>& SuperCell)
{
    TFrame *curFrame;
    
    curFrame = SuperCell.firstFrame;
    
    while(curFrame != nullptr)
    {
        out << *curFrame << std::endl;
        curFrame = curFrame->nextFrame;
    }
    
    return out;
}

} // namespace PIC
