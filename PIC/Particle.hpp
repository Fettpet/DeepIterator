#pragma once
#include <array>
#include <iostream>

namespace hzdr 
{
template<typename TPosition, uint_fast32_t dim>
struct Particle;

template<typename TPosition>
struct Particle<TPosition, 1>
{
    typedef TPosition                   Valuetype;
    typedef Particle<TPosition, 1>      particle_type;
    typedef TPosition                   position_type;
    static constexpr uint32_t Dim = 1;
    Particle():
        data({0}){}
    
    Particle(const TPosition& x):
        data({x})
    {}
    
    Particle(const Particle& par):
        data(par.data)
    {}
    
    
    template<typename TIndex>
    TPosition& 
    operator[] (const TIndex& pos)
    {
        return data[pos];
    }
    
    template<typename TIndex>
    const 
    TPosition& 
    operator[] (const TIndex& pos)
    const 
    {
        return data[pos];
    }
    
    std::array<TPosition, 1> data;
}; // struct Particle<TPosition, 1>


template<typename TPosition>
struct Particle<TPosition, 2>
{
    typedef TPosition                   ValueType;
    typedef Particle<TPosition, 2>      particle_type;
    typedef TPosition                   position_type;
    static constexpr uint32_t Dim = 2;
    
    Particle():
        data({0,0}){}
    
    Particle(const TPosition& x, const TPosition& y):
        data({x, y})
    {}
    
    Particle(const Particle& par):
        data(par.data)
    {}
    
    template<typename TIndex>
    TPosition& 
    operator[] (const TIndex& pos)
    {
        return data[pos];
    }
    
    template<typename TIndex>
    const 
    TPosition& 
    operator[] (const TIndex& pos)
    const 
    {
        return data[pos];
    }
    
    std::array<TPosition, 2> data;
}; // struct Particle<TPosition, 2>

template<typename TPos>
std::ostream& operator<<(std::ostream& out, Particle<TPos, 1> const & par)
{
    out << "(" << par[0] << ")";
    return 0;
}

template<typename TPos>
std::ostream& operator<<(std::ostream& out, Particle<TPos, 2> const & par)
{
    out << "(" << par[0] << ", " << par[1] << ")";
    return out;
}
} // namespace PIC