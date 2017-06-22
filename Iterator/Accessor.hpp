/**
 * @author Sebastian Hahn (t.hahn[at]hzdr.de) 
 * @brief The accessor handle the access to a value. It is a policy in the 
 * DeepIterator. We had a trait called \see hzdr::traits::isIndexable. It has the
 * condition that the datastructure has operator [] overloaded. If the condition
 * is satisfied, you doesnt need to implement an own Accessor. In other cases, 
 * you need to write one. 
 * The accessor has a function 
 * C* get( T*). T is the input datatype and is the output datatype. You need to 
 * declare C as ReturnType. i.e 
 * typedef C ReturnType;
 */

#pragma once
#include "PIC/Frame.hpp"
#pragma once
#include "PIC/Supercell.hpp"
#include <iostream>
#include <boost/core/ignore_unused.hpp>
#include "Definitions/hdinline.hpp"
namespace hzdr
{
class Indexable;
     
template<typename TData,
        typename SFIANE = void>
struct Accessor;

template<typename TData >
struct Accessor<TData, typename std::enable_if<traits::IsIndexable<TData>::value>::type> 
{

    
    template<typename TContainer, 
             typename TComponent,
             typename TIndex, 
             typename TRuntime>
    HDINLINE
    static
    auto 
    get(TContainer* containerPtr, 
        TComponent* componentenPtr, 
        const TIndex& pos, 
        const TRuntime& runtimeVariables)
    -> TComponent*
    {
        const auto nbElem = traits::NeedRuntimeSize<TContainer>::test(containerPtr) * runtimeVariables.getNbElements() 
                          + (1 - traits::NeedRuntimeSize<TContainer>::test(containerPtr)) * traits::NumberElements<TContainer>::value;
        if(pos < nbElem && pos >= 0)
        {
            return &((*containerPtr)[pos]); 
        }
        else 
        {
            return nullptr;
        }
        
    }
}; // Accessor< Indexable >




template<typename TFrame>
struct Accessor<SuperCell<TFrame>, void >
{
    typedef TFrame                          FrameType;
    typedef FrameType*                      FramePointer;
    typedef FrameType                       ReturnType;
    typedef ReturnType&                     ReturnReference;
    typedef ReturnType*                     ReturnPtr;

    template<typename TContainer, 
             typename TComponent,
             typename TIndex, 
             typename TRuntimeVariables>
    HDINLINE
    static
    auto 
    get(TContainer* con, TComponent* com, const TIndex& pos, const TRuntimeVariables& runtime)
    -> TComponent*
    {
        return com;
    }
    
}; // Accessor < SuperCell >

}// namespace hzdr
