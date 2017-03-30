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
    static uint_fast32_t                currentValue;
    typedef TPosition                   Valuetype;
    typedef Particle<TPosition, 1>      particle_type;
    typedef TPosition                   position_type;
    static constexpr uint32_t Dim = 1;
    Particle():
        data({++currentValue}){}
    
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
    
    inline 
    bool 
    operator==(const Particle& other)
    {
        return other.data[0] == data[0];
    }
    
    std::array<TPosition, 1> data;
}; // struct Particle<TPosition, 1>


template<typename TPosition>
struct Particle<TPosition, 2>
{
    static uint_fast32_t                currentValue;
    typedef TPosition                   ValueType;
    typedef Particle<TPosition, 2>      particle_type;
    typedef TPosition                   position_type;
    static constexpr uint32_t Dim = 2;
    
    Particle():
        data({++currentValue, ++currentValue}){}
    
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
    
    inline 
    bool 
    operator==(const Particle& other)
    const
    {
        return other.data[0] == data[0] and other.data[1] == data[1];
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
template<typename TPos>
uint_fast32_t Particle<TPos, 2>::currentValue = 0;

template<typename TPos>
uint_fast32_t Particle<TPos, 1>::currentValue = 0;
} // namespace PIC