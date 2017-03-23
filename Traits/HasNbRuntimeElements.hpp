#pragma once


/**
 * @brief check whether a class contains the member nbRuntimeElements
 * 
 */
namespace hzdr 
{
namespace traits 
{
template<typename T> 
struct HasNbRuntimeElementst { 
    struct Fallback { uint_fast32_t nbRuntimeElements; };
    struct Derived : T, Fallback { };

    template<typename C, C> struct ChT; 

    template<typename C> static char (&f(ChT<uint_fast32_t Fallback::*, &C::nbRuntimeElements>*))[1]; 
    template<typename C> static char (&f(...))[2]; 

    static bool const value = sizeof(f<Derived>(0)) == 2;
};  // HasOffset
}//traits
}// hzdr
