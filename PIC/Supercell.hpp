/**
 * @author Sebastian Hahn
 * @brief A PIConGPU like datastructure. The supercell contains some frames.
 * The frames are in a linked list. Each frame has the pointer nextFrame and 
 * previousFrame. Only the last frame is not full with particles. The supercell
 * stores the number of particles in the last frame. Each supercell has two 
 * pointers to frame: firstFrame and lastFrame.
 * 
 */
#pragma once
#include <iostream>
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
        lastFrame(nullptr),
        nbParticlesInLastFrame(0)
    {}
    
    SuperCell& operator=(const SuperCell&) = default;
    /**
     * @param nbFrames: number of frames within the supercell,
     * @param nbParticle number of particles in the last frame
     */
    HDINLINE
    SuperCell(uint32_t nbFrames, uint32_t nbParticles):
        firstFrame(new TFrame()),
        nbParticlesInLastFrame(nbParticles)
    {
        TFrame *curFrame;
        curFrame = firstFrame;
        for(uint32_t i=1; i<nbFrames; ++i)
        {
            curFrame->nextFrame = new TFrame();
            curFrame->nextFrame->previousFrame = curFrame;
            curFrame = curFrame->nextFrame;
        }
        
        lastFrame = curFrame;
        
        for(uint32_t i=nbParticles; i<TFrame::nbParticleInFrame; ++i)
        {
            for(uint32_t dim=0; dim < TFrame::Dim; ++dim)
                lastFrame->particles[i].data[dim] = -1;
        }
        
    }
    
    TFrame *firstFrame, *lastFrame;
    uint32_t nbParticlesInLastFrame;
}; // struct SuperCell

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
