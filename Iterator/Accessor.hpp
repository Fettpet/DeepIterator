/**
 * \struct Accessor
 * 
 * @author Sebastian Hahn (t.hahn[at]hzdr.de) 
 * 
 * @brief The accessor handle the access to a value. It is a policy in the 
 * DeepIterator. 
 * 
 * We had a trait called IsIndexable. It has the condition that the datastructure 
 * has operator [] overloaded. If the condition
 * is satisfied, you doesnt need to implement an own Accessor. In other cases, 
 * you need to write one.
 * The accessor has a function 
 * <b>C* get( T*).</b> T is the input datatype and is the output datatype. You need to 
 * declare C as ReturnType. i.e 
 * <b>typedef C ReturnType;</b>
 * 
 * @tparam TContainer The container over which you like to iterate. 
 * 
 * @tparam SFIANE Used intern for sfiane.
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
     
template<typename TContainer,
        typename SFIANE = void>
struct Accessor;

/**
 * @brief This specialication implements the case, that the Container is indexable.
 * @tparam The container over which you like to iterate. 
 */
template<typename TContainer >
struct Accessor<TContainer, typename std::enable_if<traits::IsIndexable<TContainer>::value>::type> 
{

    
    template<typename TComponent,
             typename TIndex>
    HDINLINE
    auto 
    get(TContainer* containerPtr, 
        TComponent* componentenPtr, 
        const TIndex& pos)
    const
    -> TComponent*
    
    {
        typedef traits::NumberElements< TContainer> NbElem;
        NbElem nbElem;        

        if(pos < nbElem.size(*containerPtr) && pos >= 0)
        {
            return &((*containerPtr)[pos]); 
        }
        else 
        {
            return nullptr;
        }
        
    }
}; // Accessor< Indexable >



/**
 * @brief Frames in supercells are not indexable. This specialication implements
 * the access to a frame within a supercell
 */
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
             typename TIndex>
    HDINLINE
    auto 
    get(TContainer* con, TComponent* com, const TIndex& pos)
    const
    -> TComponent*
    {
        return com;
    }
    
}; // Accessor < SuperCell >

}// namespace hzdr
