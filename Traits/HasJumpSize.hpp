#pragma once


/**
 * @brief check whether a class contains the member jumpsize
 * 
 */
namespace hzdr 
{
namespace traits 
{
template<typename T> 
struct HasJumpsize { 
    struct Fallback { int_fast32_t jumpsize; }; // introduce member name "x"
    struct Derived : T, Fallback { };

    template<typename C, C> struct ChT; 

    template<typename C> static char (&f(ChT<int_fast32_t Fallback::*, &C::jumpsize>*))[1]; 
    template<typename C> static char (&f(...))[2]; 

    static bool const value = sizeof(f<Derived>(0)) == 2;
};  // HasJumpsize
}//traits
}// hzdr
