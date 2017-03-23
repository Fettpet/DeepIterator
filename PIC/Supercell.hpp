#pragma once
#include <iostream>


namespace hzdr
{
template<typename TFrame>
struct SuperCell
{

    typedef TFrame frame_type;
    typedef TFrame FrameType;
    typedef TFrame ValueType;
    /**
     * @param nbFrames: number of frames within the supercell,
     * @param nbParticle number of particles in the last frame
     */
    SuperCell(uint32_t nbFrames, uint32_t nbParticles):
        firstFrame(new TFrame(0)),
        nbParticlesInLastFrame(nbParticles)
    {
        TFrame *curFrame;
        curFrame = firstFrame;
        for(uint32_t i=1; i<nbFrames; ++i)
        {
            curFrame->nextFrame = new TFrame(0);
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
std::ostream& operator<<(std::ostream& out, const SuperCell<TFrame>& SuperCell)
{
    TFrame *curFrame;
    
    curFrame = SuperCell.firstFrame;
    
    while(curFrame != nullptr)
    {
        std::cout << *curFrame << std::endl;
        curFrame = curFrame->nextFrame;
    }
    
    return out;
}

} // namespace PIC