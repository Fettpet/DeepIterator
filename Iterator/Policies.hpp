/**
 * 
 * 
 * 
 */



#pragma once
#include "Definitions/hdinline.hpp"
namespace hzdr 
{
/**
 * @brief This policy is used to decide in which direction the iterator walk 
 * through the datastructure.
 * 
 * This is importend for our implementation of the navigator. We had two cases 
 * implemented:
 * Forward: start at the first entry and go to the last one.
 * Backward: start at the last entry and go to the first one
 * 
 * \see Navigator
 */



/**
 * @brief This class gets a value with the constructor. The value is returned 
 * with the operator ().
 */
template<typename T, unsigned value = 0>
struct SelfValue;


template<typename T, unsigned value>
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
    T 
    operator() ()
    const 
    {
        return value;
    }
};


template<typename T>
struct SelfValue<T, 0>
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
};

/**
 * @brief The NoChild is used to define the last layer in a nested Iterator
 */
struct NoChild {};
    
}

