/* Copyright 2018 Sebastian Hahn

 * This file is part of DeepIterator.
 *
 * DeepIterator is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DeepIterator is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DeepIterator.
 * If not, see <http://www.gnu.org/licenses/>.
 */

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
