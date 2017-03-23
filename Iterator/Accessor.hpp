/**
 * @author Sebastian Hahn (t.hahn[at]hzdr.de) 
 * @brief The accessor handle the access to a value. It is a policy in the 
 * DeepIterator. We had a trait called \see hzdr::traits::isIndexable. It has the
 * condition that the datastructure has operator [] overloaded. If the condition
 * is satisfied, you doesnt need to implement an own Accessor. In other cases, 
 * you need to write one. 
 * 
 */

#pragma once
#include "PIC/Frame.hpp"
#pragma once
#include "PIC/Supercell.hpp"
#include <iostream>
#include <boost/core/ignore_unused.hpp>
namespace hzdr
{
class Indexable;
     
template<typename TData>
struct Accessor;

template<>
struct Accessor<Indexable> 
{
    
    template<typename TContainer, typename TIndex>
    static
    auto 
    get(TContainer& con, const TIndex& pos)
    -> typename TContainer::ValueType*
    {
        return &(con[pos]);
    }
    
    
    template<typename TContainer, typename TIndex>
    static
    auto 
    get(TContainer* con, const TIndex& pos)
    -> typename TContainer::ValueType*
    {
        return &((*con)[pos]);
    }
}; // Accessor< Indexable >




template<typename TFrame>
struct Accessor<SuperCell<TFrame> >
{
    typedef TFrame                          FrameType;
    typedef FrameType*                      FramePointer;
    typedef FrameType                       ReturnType;
    typedef ReturnType&                     ReturnReference;
    typedef ReturnType*                     ReturnPtr;
    
    Accessor() = default;
    

    static
    ReturnPtr
    inline
    get(FramePointer frame)
    {
        return frame;
    }
    
    

    static
    ReturnPtr
    inline
    get(FrameType& frame)
    {
        return &frame;
    }
 
}; // Accessor < SuperCell >

}// namespace hzdr