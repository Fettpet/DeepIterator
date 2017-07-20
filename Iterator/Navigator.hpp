/**
 * \struct Navigator
 * @author Sebastian Hahn (t.hahn< at >hzdr.de )
 * 
 * @brief The navigator is used to get the first element, the next element and
 * a decision function, wheter the end is reached or not. 
 * 
 * @tparam TContainer The datatype of the datastructure. If the datastructure is
 * indexable you doesnt need to write your own navigator. 
 * It has three templates:
 * 
 * @tparam Direction There are two possibilities: Forward: The iterator start at
 * the first element and go to the last one; Backward: The iterator start at the 
 * last element and go to the first one
 * 
 * @tparam SFIANE used for SFIANE
 * 
 * The navigator has three function: One is used to get the entry point to an 
 * container, the second gives the next component and the last one decides wheter
 * the end is reached. The header of these functions are:
 *  void first(TContainer* conPtrIn, TContainer*& conPtrOut, TComponent*& compontPtr, TIndex& index, const TOffset& offset) const
    void next(TContainer* ptr, TComponent* elem, TIndex& index, const TJumpsize& jump) const
    bool isEnd(TContainer const * const containerPtr, TComponent const * const compontPtr, const TIndex& index, const TJumpsize& jumpsize)
 * The attributs function first has five parameter. The first one is a pointer to
 the container given by the constructor of the DeepIterator. The second parameter
 is pointer to the container, stored within the iterator. The third paramter is a
 pointer to the current component. The DeepIterator use the index paramter to 
 decide the position of the component within the container. The last paramter is
 the offset, for parallel applications. The parameter conPtrOut, componentPtr and
 index are the output of this function. 
 The parameter for the second and third function are similar. The difference are: 
 1. There is no input container pointer and 2. the offset is replaced by the jumpsize. 
 */

#pragma once
#include "Policies.hpp"
#include "PIC/Frame.hpp"
#include <boost/core/ignore_unused.hpp>
#include "PIC/Supercell.hpp"
#include <type_traits>
#include "Definitions/hdinline.hpp"
#include "Traits/Componenttype.hpp"
#include "Traits/IsIndexable.hpp"
namespace hzdr 
{

    class Indexable;

template<typename TContainer,
         typename TDirection,
         typename SFIANE = void>
struct Navigator;
    
/** ****************
 * @brief This one is used for indexable
 * datatypes. It started at the last element and go to the first one.
 *****************/
template<typename TContainer, uint_fast32_t jumpSize>
struct Navigator<TContainer,
                 hzdr::Direction::Backward<jumpSize>,
                 typename std::enable_if<traits::IsIndexable<TContainer>::value>::type >
{
public:
 
    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TIndex, 
             typename TComponent>
    HDINLINE
    
    void 
    next(TContainer* ptr, 
         TComponent* elem,
         TIndex& index)
    const
    {
        index -= jumper.getJumpsize();
    }
    
    template<typename TOffset,
            typename TComponent,
            typename TIndex>
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index, 
          const TOffset& offset)
    const
    {
        typedef traits::NumberElements< TContainer> NbElem;
        NbElem nbElem;   
        conPtrOut = conPtrIn;
        index = nbElem.size(*conPtrOut) - 1 - offset;
    }
    
    template<typename TComponent,
             typename TIndex>
    
    bool 
    HDINLINE 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const compontPtr,
          const TIndex& index)
    const
    {
        typedef traits::NumberElements< TContainer> NbElem;
        NbElem nbElem;       
        return static_cast<int_fast32_t>(index) < -1  * ((static_cast<int_fast32_t>(jumper.getJumpsize() -  (nbElem.size(*containerPtr) % jumper.getJumpsize()) %jumper.getJumpsize())));
    }
    hzdr::Direction::Backward<jumpSize> jumper;
}; // Navigator<Forward, Frame, jumpSize>
    
    
/** *******************
 * @brief This one is used for indexable
 * datatypes. The direction is forwars i.e. is starts at the first element and
 * go to the last one.
 *****************/////    
template<typename TContainer, uint_fast32_t jumpSize>
struct Navigator<TContainer,
                 hzdr::Direction::Forward<jumpSize>,
                 typename std::enable_if<traits::IsIndexable<TContainer>::value>::type > 
{
public:
    

    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TIndex, 
             typename TComponent>
    HDINLINE
    
    void 
    next(TContainer* ptr, 
         TComponent* elem,
         TIndex& index)
    const
    {
        index += jumper.getJumpsize();
        
    }
    
    template<typename TOffset,
            typename TIndex,
            typename TComponent>
    
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index, 
          const TOffset& offset)
    const
    {

        conPtrOut = conPtrIn;
        index = offset;
    }
    
    template<typename TComponent,
            typename TIndex>
    
    bool 
    HDINLINE 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const compontPtr,
          const TIndex& index)
    const
    {
        typedef traits::NumberElements< TContainer> NbElem;
        NbElem nbElem;        
        return static_cast<int_fast32_t>(index) >= static_cast<int_fast32_t>(nbElem.size(*containerPtr) + (jumper.getJumpsize() -  (nbElem.size(*containerPtr) % jumper.getJumpsize())) % jumper.getJumpsize()) ;
    }
    hzdr::Direction::Forward<jumpSize> jumper;
}; // Navigator<Backward, Frame, jumpSize>


/** ****************
 * @brief This Navigator can acess all Frames in a Supercell. The direction is
 * forward.
 *****************/

template<typename TFrame, uint_fast32_t jumpSize>
struct Navigator< 
    hzdr::SuperCell<TFrame>, 
    hzdr::Direction::Forward<jumpSize>, 
    void> 
{
    typedef hzdr::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
    
public:
    
    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
    */
    template<typename TIndex, 
             typename TContainer,  
             typename TComponent>
    HDINLINE
    
    void 
    next(TContainer*& ptr, 
         TComponent*& elem,
         TIndex& index)
    const
    {

        for(uint_fast32_t i=0; i<jumper.getJumpsize(); ++i)
        {
                             
            if(elem == nullptr) 
            {
                ptr = nullptr;
                break;
            }
            elem = elem->nextFrame;
        }
    }

    
    template<typename TOffset,
             typename TComponent,
             typename TIndex,
             typename TContainer>
    
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index, 
          const TOffset& offset)
    const
    {
         
        if(conPtrIn != nullptr)
        {
            compontPtr = conPtrIn->firstFrame;
            for(uint i=0; i < offset; ++i)
            {
                if(compontPtr == nullptr) 
                {
                    conPtrOut = nullptr;
                    break;
                }
                compontPtr = compontPtr->nextFrame;
            }
        } 
        else 
        {

            compontPtr = nullptr;
        }
            
    }
    
    template<typename TComponent,
            typename TIndex,
            typename TContainer>
    
    bool 
    HDINLINE 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const compontPtr,
          const TIndex&)
    const
    {
        return (compontPtr == nullptr) and (containerPtr == nullptr);
    }
    hzdr::Direction::Forward<jumpSize> jumper;
}; // Navigator<Forward, Frame, jumpSize>
    
    
    
/**
 * @brief this implementation use supercells. The direction is backward. 
 */    
template<typename TFrame, uint_fast32_t jumpSize>
struct Navigator< hzdr::SuperCell<TFrame>, hzdr::Direction::Backward<jumpSize>, void>
{
    typedef hzdr::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
public:
    hzdr::Direction::Backward<jumpSize> jumper;

    /**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
 */
    template<typename TIndex, 
             typename TContainer,  
             typename TJumpsize>
    HDINLINE
    
    void 
    next(TContainer*& ptr, 
         typename traits::ComponentType<TContainer>::type*& elem,
         TIndex& index, 
         const TJumpsize& jumpsize)
    const
    {

        for(size_t i=0; i<jumpsize; ++i)
        {
            
            if(elem == nullptr) 
            {
                ptr = nullptr;
                break;
            }
            elem = elem->previousFrame;
        }

    }
    
    template<typename TOffset,
             typename TIndex,
             typename TComponent,
             typename TContainer>
    
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index, 
          const TOffset& offset)
    const
    {
       
        if(conPtrIn != nullptr)
        {

            compontPtr = conPtrIn->lastFrame;
            for(auto i=0; i < offset; ++i)
            {
                if(compontPtr != nullptr)
                    compontPtr = compontPtr ->previousFrame;
            }
        }
        else
        {
            compontPtr = nullptr;
        }
    }
    

    
    template<typename TComponent,
            typename TIndex,
            typename TContainer>
    
    bool 
    HDINLINE 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const componentPtr,
          const TIndex& index)
    const
    {
        return (componentPtr == nullptr) and (containerPtr == nullptr);
    }
    
}; // Navigator<Forward, Frame, jumpSize>

}// namespace hzdr
