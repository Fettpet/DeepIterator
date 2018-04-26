#pragma once
#include "deepiterator/definitions/hdinline.hpp"
namespace deepiterator 
{

/**
 * @brief This class gets a value with the constructor. The value is returned 
 * with the operator ().
 */
template<typename T, unsigned value = 999999999u>
struct SelfValue
{
    HDINLINE
    SelfValue() = default;
    
    HDINLINE
    SelfValue(SelfValue const &) = default;
    
    HDINLINE
    SelfValue(SelfValue&&) = default;
    
    
    HDINLINE
    SelfValue& operator=( SelfValue const &) = default;
    
    HDINLINE
    constexpr
    T 
    operator() ()
    const
    noexcept
    {
        return value;
    }
} ;


template<typename T>
struct SelfValue<T, 999999999u>
{
    
    HDINLINE
    SelfValue(T const & value):
        value(value)
        {}
    
    HDINLINE
    SelfValue(SelfValue const &) = default;
    
    HDINLINE
    SelfValue(SelfValue&&) = default;
    
    HDINLINE
    SelfValue() = delete;
    
    HDINLINE
    SelfValue& operator=( SelfValue const &) = default;
    
    HDINLINE
    T 
    operator() ()
    const 
    {
        return value;
    }
protected:
    T value;
} ;

} // namespace deepiterator
