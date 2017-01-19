#pragma once
#include <iostream>
#include <iomanip>
template<
    typename TParticle,
    unsigned nbParticles>
struct Frame
{
    typedef TParticle particle_type;
    typedef typename TParticle::position_type particle_position_type;
    typedef Frame<TParticle, nbParticles> FrameType;
    
    constexpr static unsigned Dim = TParticle::Dim;
    constexpr static unsigned nbParticleInFrame = nbParticles;
    
   
    Frame(){}
    
    Frame(...):
        nextFrame(nullptr), previousFrame(nullptr)
    {
        for(auto &par: particles)
        {
           for(int i=0; i<Dim; ++i)
           {
               par.data[i] = rand() % 100;
           }
        }
        
    }
    
    FrameType& operator=(const FrameType& other)
    {
        particles = other.particles;
    }
    
    std::array<TParticle, nbParticles> particles;
    FrameType *nextFrame, *previousFrame;
};

template<
    typename TParticle,
    unsigned nbParticles>
std::ostream& operator<<(std::ostream& out, const Frame<TParticle, nbParticles>& f)
{
    out << "[";
    for(int i=0; i< nbParticles; ++i)
    {
        out << f.particles[i] << ", ";
    }
    out << "]";
    return out;
}
