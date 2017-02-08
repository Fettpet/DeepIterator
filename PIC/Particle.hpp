#pragma once
#include <array>
#include <iostream>

namespace Data 
{
template<typename TPosition, unsigned dim>
struct Particle;

template<typename TPosition>
struct Particle<TPosition, 1>
{
    typedef Particle<TPosition, 1> particle_type;
    typedef TPosition position_type;
    static constexpr unsigned Dim = 1;
    Particle():
        data({0}){}
    
    Particle(const TPosition& x):
        data({x})
    {}
    
    Particle(const Particle& par):
        data(par.data)
    {}
    
    
    
    std::array<TPosition, 1> data;
}; // struct Particle<TPosition, 1>


template<typename TPosition>
struct Particle<TPosition, 2>
{
    typedef Particle<TPosition, 2> particle_type;
    typedef TPosition position_type;
    static constexpr unsigned Dim = 2;
    
    Particle():
        data({0,0}){}
    
    Particle(const TPosition& x, const TPosition& y):
        data({x, y})
    {}
    
    Particle(const Particle& par):
        data(par.data)
    {}
    
    std::array<TPosition, 2> data;
}; // struct Particle<TPosition, 2>

template<typename TPos>
std::ostream& operator<<(std::ostream& out, Particle<TPos, 1> const & par)
{
    out << "(" << par.data[0] << ")";
    return 0;
}

template<typename TPos>
std::ostream& operator<<(std::ostream& out, Particle<TPos, 2> const & par)
{
    out << "(" << par.data[0] << ", " << par.data[1] << ")";
    return out;
}
} // namespace PIC