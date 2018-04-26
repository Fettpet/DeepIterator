/**
 * \struct Frame
 * @author Sebastian Hahn (t.hahn@deepiterator.de)
 * @brief A PIConGPU like datastructure. It represent the frame in PIC. We use
 * a std::array to store some Particles. Both, the particle type and the number
 * of particles within the frame, are compiletime variables. 
 * Each frame has a pointer to the next Frame and the previousFrame frame.
 */ 
 

#pragma once

#include <iomanip>
#include <iostream>
#include "deepiterator/definitions/hdinline.hpp"
#include "deepiterator/traits/Traits.hpp"

namespace deepiterator
{
template<
    typename TParticle,
    int_fast32_t maxParticles
>
struct Frame
{
    typedef TParticle                           ValueType;
    typedef TParticle                           ParticleType;
    typedef ParticleType*                       ParticlePointer;
    typedef ParticleType&                       ParticleReference;
    typedef Frame<TParticle, maxParticles>      FrameType;
    
    constexpr static int_fast32_t Dim = TParticle::Dim;
    constexpr static int_fast32_t maxParticlesInFrame = maxParticles;
    
    HDINLINE 
    Frame(Frame const &) = default;
    
    HDINLINE Frame(Frame &&) = default;
    
    HDINLINE
    Frame():
        
        nextFrame(nullptr), previousFrame(nullptr),
        nbParticlesInFrame(maxParticles)
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
    
    HDINLINE
    Frame(const uint_fast32_t& nbParticles):
        nextFrame(nullptr), previousFrame(nullptr),
        nbParticlesInFrame(nbParticles)
    {
        static int_fast32_t value{0};
        for(auto &par: particles)
        {
           for(int_fast32_t i=0; i<Dim; ++i)
           {
               par.data[i] = value++;
           }
        }
        for(int i=nbParticles; i<maxParticles; ++i)
        {
           for(int_fast32_t j=0; j<Dim; ++j)
           {
               particles[i].data[j] = -1;
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
        return *this;
    }

    
    TParticle particles[maxParticles];

    FrameType *nextFrame, *previousFrame;
    uint_fast32_t nbParticlesInFrame;
} ; // struct Frame


template<
    typename TParticle,
    int_fast32_t maxParticles>
std::ostream& operator<<(std::ostream& out, const Frame<TParticle, maxParticles>& f)
{
    out << "[";
    for(uint_fast32_t i=0; i< maxParticles; ++i)
    {
        out << f.particles[i] << ", ";
    }
    out << "]";
    return out;
}

// traits
namespace traits 
{

    
template<
    typename TParticle,
    int_fast32_t maxParticles>
struct IsBidirectional<Frame<TParticle, maxParticles> >
{
    static const bool value = true;
} ;



template<
    typename TParticle,
    int_fast32_t maxParticles>
struct IsRandomAccessable<Frame<TParticle, maxParticles> >
{
    static const bool value = true;
} ;


template<
    typename TParticle,
    int_fast32_t maxParticles>
struct HasConstantSize<Frame<TParticle, maxParticles> >
{
    static const bool value = false;
} ;



template<
    typename TParticle,
    int_fast32_t maxParticles>
struct ComponentType<Frame<TParticle, maxParticles> >
{
    typedef TParticle type;
} ;


template<typename TParticle, int_fast32_t nb>
struct ContainerCategory<deepiterator::Frame<TParticle, nb> >
{
    typedef deepiterator::container::categorie::ArrayLike type;
};

template<typename TParticle, int_fast32_t nb>
struct NumberElements<deepiterator::Frame<TParticle, nb> >
{
    typedef deepiterator::Frame<TParticle, nb> Frame;
    

    HDINLINE
    int_fast32_t 
    operator()( Frame const * const f)
    const
    {
        return f->nbParticlesInFrame;    
    }
} ;// struct NumberElements

} // namespace traits

}// namespace deepiterator
