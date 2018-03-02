/**
 * @author Sebastian Hahn ( t.hahn@hzdr.de )
 * 
 * @brief A PIConGPU like datastructure. It has a position_type and a dimension.
 */
#pragma once
#include <array>
#include <iomanip>
#include <iostream>
#include "deepiterator/iterator/Categorie.hpp"
#include "deepiterator/definitions/hdinline.hpp"
#include "deepiterator/traits/Traits.hpp"
namespace hzdr 
{
template<typename TPosition, int_fast32_t dim>
struct Particle;

template<typename TPosition>
struct Particle<TPosition, 1>
{
    static int_fast32_t                currentValue;
    typedef TPosition                   Valuetype;
    typedef Particle<TPosition, 1>      particle_type;
    typedef TPosition                   position_type;
    static constexpr uint32_t Dim = 1;
    
    HDINLINE
    Particle()
    {
        data[0] = ++currentValue;
    }
    
    HDINLINE
    Particle(const TPosition& x):
        data(x)
    {}
    
    HDINLINE
    Particle(const Particle& par):
        data(par.data)
    {}
    
    
    template<typename TIndex>
    HDINLINE
    TPosition& 
    operator[] (const TIndex& pos)
    {
        return data[pos];
    }
    
    template<typename TIndex>
    HDINLINE
    const 
    TPosition& 
    operator[] (const TIndex& pos)
    const 
    {
        return data[pos];
    }
    
    HDINLINE
    bool 
    operator==(const Particle& other)
    {
        return other.data[0] == data[0];
    }
    
    TPosition data[1];
} ; // struct Particle<TPosition, 1>


template<typename TPosition>
struct Particle<TPosition, 2>
{
    static int_fast32_t                currentValue;
    typedef TPosition                   ValueType;
    typedef Particle<TPosition, 2>      particle_type;
    typedef TPosition                   position_type;
    static constexpr uint32_t Dim = 2;
    
    HDINLINE
    Particle()
    {
        data[0] = 0;
        data[1] = 0;
        
    }
    
    HDINLINE
    Particle(const TPosition& x, const TPosition& y)
    {
        data[0] = x;
        data[1] = y;
    }
    
    HDINLINE
    Particle(const TPosition& x)
    {
        data[0] = x;
        data[1] = x;
    }
    
    HDINLINE
    Particle(const Particle& par)
    {
        data[0] = par.data[0];
        data[1] = par.data[1];
    }
    
    
    template<typename TIndex>
    HDINLINE
    TPosition& 
    operator[] (const TIndex& pos)
    {
        return data[pos];
    }
    
    template<typename T>
    HDINLINE
    Particle&
    operator*=(const T& m)
    {
        data[0] *= m;
        data[1] *= m;
        return *this;
    }
    
    template<typename TIndex>
    HDINLINE
    const 
    TPosition& 
    operator[] (const TIndex& pos)
    const 
    {
        return data[pos];
    }
    
    HDINLINE
    bool 
    operator==(const Particle& other)
    const
    {
        return other.data[0] == data[0] and other.data[1] == data[1];
    }
    
    HDINLINE 
    Particle&
    operator=(TPosition const & num)
    {
        data[0] = num;
        data[1] = num;
    }
    
    HDINLINE 
    Particle&
    operator++()
    {
        data[0]++;
        data[1]++;
        return *this;
    }
    TPosition data[2];

} ; // struct Particle<TPosition, 2>

// traits
namespace traits 
{



template<
    typename TProperty,
    int_fast32_t maxParticles>
struct HasConstantSize<Particle<TProperty, maxParticles> >
{
    static const bool value = true;
} ;


template<
    typename TProperty,
    int_fast32_t maxParticles>
struct ComponentType<Particle<TProperty, maxParticles> >
{
    typedef TProperty type;
} ;


template<typename TProperty, int_fast32_t nb>
struct ContainerCategory<Particle<TProperty, nb> >
{
    typedef hzdr::container::categorie::ArrayLike type;
};

template<typename TPos, int_fast32_t nb>
struct NumberElements<hzdr::Particle<TPos, nb> >
{
    typedef hzdr::Particle<TPos, nb> Particle;
    
    HDINLINE
    int_fast32_t 
    constexpr
    operator()(Particle* )
    const
    {
        return nb;    
    }
    
}; 

} // namespace traits



template<typename TPos>
HDINLINE
std::ostream& operator<<(std::ostream& out, Particle<TPos, 1> const & par)
{
    out << "(" << par[0] << ")";
    return out;
}

template<typename TPos>
HDINLINE
std::ostream& operator<<(std::ostream& out, Particle<TPos, 2> const & par)
{
    out << "(" << par[0] << ", " << par[1] << ")";
    return out;
}
template<typename TPos>
int_fast32_t Particle<TPos, 2>::currentValue = 0;

template<typename TPos>
int_fast32_t Particle<TPos, 1>::currentValue = 0;
} // namespace hzdr
