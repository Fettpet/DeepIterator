#pragma once
#include <iostream>
#include <iomanip>
namespace hzdr
{
template<
    typename TParticle,
    unsigned nbParticles>
struct Frame
{
    typedef TParticle                           ParticleType;
    typedef ParticleType*                       ParticlePointer;
    typedef ParticleType&                       ParticleReference;
    typedef typename TParticle::position_type   particle_position_type;
    typedef Frame<TParticle, nbParticles>       FrameType;
    
    constexpr static unsigned Dim = TParticle::Dim;
    constexpr static unsigned nbParticleInFrame = nbParticles;
    
   
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
    
    template<typename TIndex>
    inline
    ParticleReference
    operator[] (const TIndex& idx)
    {
        return particles[idx];
    }
    
    template<typename TIndex>
    inline
    const
    ParticleReference
    operator[] (const TIndex& idx)
    const
    {
        return particles[idx];
    }

    
    FrameType& operator=(const FrameType& other)
    {
        particles = other.particles;
    }
    
    std::array<TParticle, nbParticles> particles;
    FrameType *nextFrame, *previousFrame;
}; // struct Frame

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

}// namespace PIC