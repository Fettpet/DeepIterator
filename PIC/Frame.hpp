#pragma once
#include <iostream>
#include <iomanip>
namespace hzdr
{
template<
    typename TParticle,
    uint_fast32_t nbParticles>
struct Frame
{
    typedef TParticle                           ValueType;
    typedef TParticle                           ParticleType;
    typedef ParticleType*                       ParticlePointer;
    typedef ParticleType&                       ParticleReference;
    typedef typename TParticle::position_type   particle_position_type;
    typedef Frame<TParticle, nbParticles>       FrameType;
    
    constexpr static uint_fast32_t Dim = TParticle::Dim;
    constexpr static uint_fast32_t nbParticleInFrame = nbParticles;
    
   
    Frame(...):
        nextFrame(nullptr), previousFrame(nullptr)
    {
        for(auto &par: particles)
        {
           for(uint_fast32_t i=0; i<Dim; ++i)
           {
               par.data[i] = rand() % 100;
           }
        }
        
    }
    
    template<typename TIndex>
    inline
    ParticleReference
    operator[] (const TIndex& pos)
    {
        return particles[pos];
    }
    
    template<typename TIndex>
    inline
    const
    ParticleReference
    operator[] (const TIndex& pos)
    const
    {
        return particles[pos];
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
    uint_fast32_t nbParticles>
std::ostream& operator<<(std::ostream& out, const Frame<TParticle, nbParticles>& f)
{
    out << "[";
    for(uint_fast32_t i=0; i< nbParticles; ++i)
    {
        out << f.particles[i] << ", ";
    }
    out << "]";
    return out;
}

}// namespace PIC