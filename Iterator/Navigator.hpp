/**
 * @author Sebastian Hahn (t.hahn<at>hzdr.de)
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
         uint_fast32_t jumpSize=0>
struct Navigator;
    
/** ****************
 * @brief The first implementation of the Navigator. This one is used for indexable
 * datatypes. It started at the last element and go to the first one.
 *****************/
template<uint_fast32_t jumpSize>
struct Navigator<Indexable,
                 hzdr::Direction::Backward, 
                 jumpSize>
{
public:
    
/**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
 */
    template<typename TIndex, 
             typename TContainer,
             uint_fast32_t size=jumpSize,
             typename TRuntimeVariables>
    static
    typename std::enable_if<size != 0>::type
    inline
    next(TContainer* ptr, TIndex& index, const TRuntimeVariables& runtime) 
    {
        boost::ignore_unused(runtime);
        index -= jumpSize;
    }
    
    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TIndex, 
             typename TContainer,  
             typename TRuntimeVariables,
             uint_fast32_t size=jumpSize>
    inline
    static
    typename std::enable_if<size == 0>::type
    next(TContainer* ptr, 
         TIndex& index, 
         const TRuntimeVariables& run)
    
    {
        index -= run.jumpsize;
    }
    
    template<typename TRuntimeVariables>
    static
    uint_fast32_t 
    inline 
    first( const TRuntimeVariables& runtime)
    {
        return runtime.nbRuntimeElements - 1 - runtime.offset;
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
    
/** *******************
 * @brief the second implementation of the navigator. This one is used for indexable
 * datatypes. The direction is forwars i.e. is starts at the first element and
 * go to the last one.
 *
 *****************/////    
template<uint_fast32_t jumpSize>
struct Navigator<Indexable,
                 hzdr::Direction::Forward, 
                 jumpSize>
{
public:
    
/**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
 */
    template<typename TIndex, typename TContainer, uint_fast32_t size=jumpSize>
    static
    typename std::enable_if<size != 0>::type
    inline
    next(TContainer* ptr, TIndex& index) 
    {
        index += jumpSize;
    }
    
    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TIndex, 
             typename TContainer,  
             typename TRuntimeVariables,
             uint_fast32_t size=jumpSize>
    inline
    static
    typename std::enable_if<size == 0>::type
    next(TContainer* ptr, 
         TIndex& index, 
         const TRuntimeVariables& run)
    
    {

        index += run.jumpsize;
    }
    
    template<typename TRuntimeVariables>
    static
    uint_fast32_t 
    inline 
    first(const TRuntimeVariables& run)
    {
        return run.offset;
    }
    
}; // Navigator<Backward, Frame, jumpSize>


/** ****************
 * @brief This Navigator can acess all Frames in a Supercell. The direction is
 * forward.
 *****************/

template<typename TFrame,
         uint_fast32_t jumpSize>
struct Navigator< hzdr::SuperCell<TFrame>, hzdr::Direction::Forward, jumpSize>
{
    typedef hzdr::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
    
public:
    
    /**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
template<typename TIndex, typename TContainer, uint_fast32_t jumps = jumpSize>
    static
    typename std::enable_if<jumps!=0>::type 
    inline
    next(TContainer*& ptr, TIndex& index) 
    {
        for(size_t i=0; i<jumpSize; ++i)
        {
            
            if(ptr == nullptr) break;
            ptr = ptr->nextFrame;
        }
    }
     */
    
    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TContainer, typename TRuntime>
    static
    void
    inline
    next(TContainer*& ptr, const TRuntime& runtimeVariables) 
    {
        for(size_t i=0; i<runtimeVariables.jumpsize; ++i)
        {
            
            if(ptr == nullptr) break;
            ptr = ptr->nextFrame;
        }
    }

    
    template< typename TRuntime>
    static 
    FramePointer
    inline
    first(const SuperCellType* supercell, const TRuntime& runtimeVariables)
    {
        if(supercell != nullptr)
        {
            auto ptr = supercell->firstFrame;
            for(auto i=0; i < runtimeVariables.offset; ++i)
            {
                ptr = ptr->nextFrame;
            }
            return ptr;
        }
        return nullptr;
    }
    
    static 
    FramePointer
    inline
    first(nullptr_t)
    {
        return nullptr;
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
    
    
/**
 * @brief this implementation use supercells. The direction is backward. 
 */    
template<typename TFrame, uint_fast32_t jumpSize>
struct Navigator< hzdr::SuperCell<TFrame>, hzdr::Direction::Backward, jumpSize>
{
    typedef hzdr::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
public:
    

    /**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
 */
    template<typename TIndex, typename TContainer, typename TRuntime, uint_fast32_t size = jumpSize>
    static
    typename std::enable_if<size != 0>::type
    inline
    next(TContainer* ptr, TIndex& index , const TRuntime& runtimeVariables) 
    {
        boost::ignore_unused(runtimeVariables);
        for(size_t i=0; i<jumpSize; ++i)
        {
            
            if(ptr == nullptr) break;
            ptr = ptr->previousFrame;
        }
    }
    
    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TIndex, typename TContainer, typename TRuntime, uint_fast32_t size = jumpSize>
    static
    typename std::enable_if<size == 0>::type
    inline
    next(TContainer* ptr, TIndex& index, const TRuntime& run) 
    {
        boost::ignore_unused(run);
        for(size_t i=0; i<jumpSize; ++i)
        {
            
            if(ptr == nullptr) break;
            ptr = ptr->previousFrame;
        }
    }
    
    template<typename TRuntime>
    static 
    FramePointer
  
    first(const SuperCellType* supercell, const TRuntime& run)
    {
        if(supercell != nullptr)
        {
            auto ptr = supercell->lastFrame;
            for(auto i=0; i < run.offset; ++i)
            {
                ptr = ptr->previousFrame;
            }
            return ptr;
        }
        return nullptr;
    }
    
    static 
    FramePointer
    first(nullptr_t)
    {
        return nullptr;
    }
}; // Navigator<Forward, Frame, jumpSize>

}// namespace hzdr