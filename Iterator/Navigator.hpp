/**
 * @author Sebastian Hahn (t.hahn< at >hzdr.de )
 * @brief The navigator is used to go to the next element. It has three templates:
 * 1. TData: The datatype of the datastructure. If the datastructure is indexable
 * you doesnt need to write your own navigator. TData must have a Valutype typename.
 * 2. Direction: There are two possibilities: 
 *      Forward: The iterator start at the first element and go to the last one
 *      Backward: The iterator start at the last element and go to the first one
 * 3. jumpsize: spezify what the next element is. There are more possibilities:
 *      1. jumpsize == 0: The jumpsize is not known at compiletime. you need to
 *          specify the jumpsize at runtime
 *      2. jumpsize == 1: go over all elements within the datastructure
 *      3. jumpsize == c>1: overjump c-1 elemnts
 * The navigator has two function:
 *  void next( TData*, const RuntimeVariables&): specify how the next element is
 *    found. the pointer currelem is set to the next element. The RuntimeVariables
 *    have three members: jumpsize, nbRuntimeElements, offset. \see RuntimeTuple.hpp
 *  TData::Valuetype* first(TData*, RuntimeVariables): specify how the first element
 *    in the datastructure is found. The first element is returned.
 * 
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
/**
 * @brief The navigator is used to go to the next element
 * It has a runtime and a compiletime implementation
 * 
 */
template<typename TData,
         hzdr::Direction TDirection,
         typename SFIANE = void>
struct Navigator;
    
/** ****************
 * @brief The first implementation of the Navigator. This one is used for indexable
 * datatypes. It started at the last element and go to the first one.
 *****************/
template<typename TData>
struct Navigator<TData,
                 hzdr::Direction::Backward,
                 typename std::enable_if<traits::IsIndexable<TData>::value>::type >
{
public:
 
    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TIndex, 
             typename TContainer,  
             typename TComponent,
             typename TRuntimeVariables>
    HDINLINE
    static
    void 
    next(TContainer* ptr, 
         TComponent* elem,
         TIndex& index, 
         const TRuntimeVariables& run)
    
    {

        index -= run.getJumpsize();
    }
    
    template<typename TRuntimeVariables,
            typename TComponent,
            typename TIndex,
            typename TContainer>
    static
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index, 
          const TRuntimeVariables& run)
    {
        conPtrOut = conPtrIn;
        index = run.getNbElements() - 1 - run.getOffset();
    }
    
    template<typename TRuntimeVariables,
            typename TComponent,
            typename TIndex,
            typename TContainer>
    static
    bool 
    HDINLINE 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const compontPtr,
          const TIndex& index, 
          const TRuntimeVariables& run)
    {
        const int_fast32_t elem = traits::NeedRuntimeSize<TContainer>::test(containerPtr)? run.getNbElements()  : traits::NumberElements< TContainer>::value;

        return index < -1  * ((run.getJumpsize() -  (elem % run.getJumpsize())) %run.getJumpsize());
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
    
/** *******************
 * @brief the second implementation of the navigator. This one is used for indexable
 * datatypes. The direction is forwars i.e. is starts at the first element and
 * go to the last one.
 *
 *****************/////    
template<typename TData>
struct Navigator<TData,
                 hzdr::Direction::Forward,
                 typename std::enable_if<traits::IsIndexable<TData>::value>::type > 
{
public:
    

    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TIndex, 
             typename TContainer,  
             typename TComponent,
             typename TRuntimeVariables>
    HDINLINE
    static
    void 
    next(TContainer* ptr, 
         TComponent* elem,
         TIndex& index, 
         const TRuntimeVariables& run)
    {
        index += run.getJumpsize();
        
    }
    
    template<typename TRuntimeVariables,
            typename TIndex,
            typename TComponent,
            typename TContainer>
    static
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index, 
          const TRuntimeVariables& run)
    {

        conPtrOut = conPtrIn;
        index =run.getOffset();
    }
    
    template<typename TRuntimeVariables,
            typename TComponent,
            typename TIndex,
            typename TContainer>
    static
    bool 
    HDINLINE 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const compontPtr,
          const TIndex& index, 
          const TRuntimeVariables& run)
    {
        const int_fast32_t elem = traits::NeedRuntimeSize<TContainer>::test(containerPtr)? run.getNbElements()  : traits::NumberElements< TContainer>::value;

        return index >= elem + ((run.getJumpsize() -  (elem % run.getJumpsize())) %run.getJumpsize()) ;
    }
    
}; // Navigator<Backward, Frame, jumpSize>


/** ****************
 * @brief This Navigator can acess all Frames in a Supercell. The direction is
 * forward.
 *****************/

template<typename TFrame>
struct Navigator< 
    hzdr::SuperCell<TFrame>, 
    hzdr::Direction::Forward, 
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
             typename TComponent,
             typename TRuntimeVariables>
    HDINLINE
    static
    void 
    next(TContainer*& ptr, 
         TComponent*& elem,
         TIndex& index, 
         const TRuntimeVariables& run)
    {

        for(int_fast32_t i=0; i<run.getJumpsize(); ++i)
        {
                             
            if(elem == nullptr) 
            {
                ptr = nullptr;
                break;
            }
            elem = elem->nextFrame;
        }
    }

    
    template<typename TRuntimeVariables,
             typename TComponent,
             typename TIndex,
             typename TContainer>
    static
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index, 
          const TRuntimeVariables& run)
    {
         
        if(conPtrIn != nullptr)
        {
            compontPtr = conPtrIn->firstFrame;
            for(uint i=0; i < run.getOffset(); ++i)
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
    
    template<typename TRuntimeVariables,
            typename TComponent,
            typename TIndex,
            typename TContainer>
    static
    bool 
    HDINLINE 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const compontPtr,
          const TIndex& index, 
          const TRuntimeVariables& run)
    {
        return (compontPtr == nullptr) and (containerPtr == nullptr);
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
    
    
/**
 * @brief this implementation use supercells. The direction is backward. 
 */    
template<typename TFrame>
struct Navigator< hzdr::SuperCell<TFrame>, hzdr::Direction::Backward, void>
{
    typedef hzdr::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
public:
    

    /**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
 */
    template<typename TIndex, 
             typename TContainer,  
             typename TRuntimeVariables>
    HDINLINE
    static
    void 
    next(TContainer*& ptr, 
         typename traits::ComponentType<TContainer>::type*& elem,
         TIndex& index, 
         const TRuntimeVariables& run)
    {

        for(size_t i=0; i<run.getJumpsize(); ++i)
        {
            
            if(elem == nullptr) 
            {
                ptr = nullptr;
                break;
            }
            elem = elem->previousFrame;
        }

    }
    
    template<typename TRuntimeVariables,
             typename TIndex,
             typename TComponent,
             typename TContainer>
    static
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index, 
          const TRuntimeVariables& run)
    {
       
        if(conPtrIn != nullptr)
        {

            compontPtr = conPtrIn->lastFrame;
            for(auto i=0; i < run.getOffset(); ++i)
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
    

    
    template<typename TRuntimeVariables,
            typename TComponent,
            typename TIndex,
            typename TContainer>
    static
    bool 
    HDINLINE 
    isEnd(TContainer const * const containerPtr,
          TComponent const * const componentPtr,
          const TIndex& index, 
          const TRuntimeVariables& run)
    {
        return (componentPtr == nullptr) and (containerPtr == nullptr);
    }
    
}; // Navigator<Forward, Frame, jumpSize>

}// namespace hzdr
