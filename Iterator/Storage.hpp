/**
 * 
 */

#pragma once
#include "PIC/Frame.hpp"
#include "PIC/Supercell.hpp"
#include <memory>

namespace hzdr 
{
enum class Index {WithIndex, WithoutIndex};
/**
 * @tparam TElement The type of the object, where you like to iterate through.
 * It needs to be specialised!
 * It needs a overloaded operator!=
 */
template<typename TElement, Index>
struct Storage;


/** ********************
 *  @brief specialisation for Frames
 *////////////////////
template<typename TElement>
struct Storage<TElement, Index::WithIndex>
{
public:
    typedef TElement                                        Type;
    typedef Type*                                           TypePointer;
    typedef Type&                                           TypeReference;
    typedef Storage<TElement, Index::WithIndex>             ThisType;

/**
 * 
 */
    Storage(TypeReference Frame):
        ptr(&Frame)
    {}
    
    bool
    operator!=(const ThisType& other)
    const
    {
        return index < other.index;
    }

        
    bool
    operator!=(nullptr_t)
    const
    {
        return true;
    }
    
    TypePointer ptr;
    size_t index;
};



template<typename TElement>
struct Storage<TElement, Index::WithoutIndex>
{
public:
    typedef TElement                                        Type;
    typedef Type*                                           TypePointer;
    typedef Type&                                           TypeReference;
    typedef Storage<TElement, Index::WithoutIndex>          ThisType;
    
    bool
    operator!=(const ThisType& other)
    const
    {
        return ptr != other.ptr;
    }

        
    bool
    operator!=(nullptr_t)
    const
    {
        return true;
    }
    
    TypePointer ptr;
};


}