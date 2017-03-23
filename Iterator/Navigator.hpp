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
 * @brief This Navigator can acess all Particles in a Frame
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
    template<typename TIndex, typename TContainer, uint_fast32_t size=jumpSize>
    static
    typename std::enable_if<size != 0>::type
    inline
    next(TContainer* ptr, TIndex& index) 
    {
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
    
    static
    uint_fast32_t 
    inline 
    first(const uint_fast32_t& offset, const uint_fast32_t& nbParticleInFrame)
    {
        return nbParticleInFrame - 1 - offset;
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
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
    
    static
    uint_fast32_t 
    inline 
    first(const uint_fast32_t& offset, const uint_fast32_t& nbParticleInFrame)
    {
        boost::ignore_unused(nbParticleInFrame);
        return offset;
    }
    
}; // Navigator<Backward, Frame, jumpSize>


/** ****************
 * @brief This Navigator can acess all Frames in a Supercell
 * Das Problem: Er gibt mir einen SuperzellenPointer rein: Ich benötige 
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

    
    static 
    FramePointer
    inline
    first(const SuperCellType* supercell)
    {
        return supercell->firstFrame;
    }
    
    static 
    FramePointer
    inline
    first(nullptr_t)
    {
        return nullptr;
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
    
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
    template<typename TIndex, typename TContainer>
    static
    void 
    inline
    next(TContainer* ptr, TIndex& index, typename std::enable_if<jumpSize!=0, uint_fast32_t >::type* = nullptr) 
    {
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
    template<typename TIndex, typename TContainer>
    static
    void 
    inline
    next(TContainer* ptr, TIndex& index, typename std::enable_if<jumpSize==0, uint_fast32_t>::type jump) 
    {
        for(size_t i=0; i<jumpSize; ++i)
        {
            
            if(ptr == nullptr) break;
            ptr = ptr->previousFrame;
        }
    }
    
    static 
    FramePointer
  
    first(const SuperCellType* supercell)
    {
        if(supercell != nullptr)
        {
            return supercell->lastFrame;
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