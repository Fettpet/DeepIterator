#pragma once
#include <iostream>


namespace Data
{
template<typename TFrame>
struct SuperCell
{

    typedef TFrame frame_type;
    
    SuperCell(unsigned nbFrames, unsigned nbParticles):
        firstFrame(new TFrame(0)),
        nbParticlesInLastFrame(nbParticles)
    {
        TFrame *curFrame;
        curFrame = firstFrame;
        for(int i=1; i<nbFrames; ++i)
        {
            curFrame->nextFrame = new TFrame(0);
            curFrame->nextFrame->previousFrame = curFrame;
            curFrame = curFrame->nextFrame;
        }
        
        lastFrame = curFrame;
        
        for(int i=nbParticles; i<TFrame::nbParticleInFrame; ++i)
        {
            for(int dim=0; dim < TFrame::Dim; ++dim)
                lastFrame->particles[i].data[dim] = -1;
        }
        
    }
    
    TFrame *firstFrame, *lastFrame;
    unsigned nbParticlesInLastFrame;
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