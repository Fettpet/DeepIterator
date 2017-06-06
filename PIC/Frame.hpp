/**
 * @author Sebastian Hahn (t.hahn@hzdr.de)
 * @brief A PIConGPU like datastructure. It represent the frame in PIC. We use
 * a std::array to store some Particles. Both, the particle type and the number
 * of particles within the frame, are compiletime variables. 
 * Each frame has a pointer to the next and the previous frame.
 */ 
 

#pragma once
#include "Definitions/hdinline.hpp"
#include <iostream>
#include <iomanip>
namespace hzdr
{
template<
    typename TParticle,
    int_fast32_t nbParticles>
struct Frame
{
    typedef TParticle                           ValueType;
    typedef TParticle                           ParticleType;
    typedef ParticleType*                       ParticlePointer;
    typedef ParticleType&                       ParticleReference;
    typedef Frame<TParticle, nbParticles>       FrameType;
    
    constexpr static int_fast32_t Dim = TParticle::Dim;
    constexpr static int_fast32_t nbParticleInFrame = nbParticles;
    
    HDINLINE
    Frame():
        nextFrame(nullptr), previousFrame(nullptr)
    {
        static int_fast32_t value{0};
        for(auto &par: particles)
        {
           for(int_fast32_t i=0; i<Dim; ++i)
           {
               par.data[i] = value++;
           }
        }
        
    }
    
    template<typename TIndex>
    HDINLINE
    ParticleReference
    operator[] (const TIndex& pos)
    {
        return particles[pos];
    }
    int sum() const
    {
        int result{0};
        
        for(auto par: particles)
        {
            result += par.data[0];
        }
        
        return result;
    }
    
    template<typename TIndex>
    HDINLINE
    ParticleReference
    operator[] (const TIndex& pos)
    const
    {
        return particles[pos];
    }

    HDINLINE
    FrameType& 
    operator=(const FrameType& other)
    {
        particles = other.particles;
    }
    
    TParticle particles[nbParticles];

    FrameType *nextFrame, *previousFrame;
}; // struct Frame

template<
    typename TParticle,
    int_fast32_t nbParticles>

std::ostream& operator<<(std::ostream& out, const Frame<TParticle, nbParticles>& f)
{
    out << "[";
    for(int_fast32_t i=0; i< nbParticles; ++i)
    {
        out << f.particles[i] << ", ";
    }
    out << "]";
    return out;
}

}// namespace PIC
