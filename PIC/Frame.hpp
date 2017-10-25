/**
 * \struct Frame
 * @author Sebastian Hahn (t.hahn@hzdr.de)
 * @brief A PIConGPU like datastructure. It represent the frame in PIC. We use
 * a std::array to store some Particles. Both, the particle type and the number
 * of particles within the frame, are compiletime variables. 
 * Each frame has a pointer to the next Frame and the previousFrame frame.
 */ 
 

#pragma once
#include "Definitions/hdinline.hpp"
#include <iostream>
#include <iomanip>
#include "Iterator/Categorie.hpp"
#include "Traits/Componenttype.hpp"
#include "Traits/IndexType.hpp"
#include "Traits/IsRandomAccessable.hpp"
#include "Traits/IsBidirectional.hpp"
#include "Traits/HasConstantSize.hpp"
namespace hzdr
{
template<
    typename TParticle,
    int_fast32_t maxParticles>
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
}; // struct Frame


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
};



template<
    typename TParticle,
    int_fast32_t maxParticles>
struct IsRandomAccessable<Frame<TParticle, maxParticles> >
{
    static const bool value = true;
};


template<
    typename TParticle,
    int_fast32_t maxParticles>
struct HasConstantSize<Frame<TParticle, maxParticles> >
{
    static const bool value = false;
};



template<
    typename TParticle,
    int_fast32_t maxParticles>
struct ComponentType<Frame<TParticle, maxParticles> >
{
    typedef TParticle type;
};


template<typename TParticle, int_fast32_t nb>
struct ContainerCategory<hzdr::Frame<TParticle, nb> >
{
    typedef hzdr::container::categorie::ArrayLike type;
};

} // namespace traits

}// namespace hzdr
