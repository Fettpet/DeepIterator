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
         int_fast32_t jumpSize=0>
struct Navigator;
    
/** ****************
 * @brief The first implementation of the Navigator. This one is used for indexable
 * datatypes. It started at the last element and go to the first one.
 *****************/
template<int_fast32_t jumpSize>
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
             int_fast32_t size=jumpSize,
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
             int_fast32_t size=jumpSize>
    inline
    static
    typename std::enable_if<size == 0>::type
    next(TContainer* ptr, 
         TIndex& index, 
         const TRuntimeVariables& run)
    
    {
        index -= run.getJumpsize();
    }
    
    template<typename TRuntimeVariables>
    static
    int_fast32_t 
    inline 
    first( const TRuntimeVariables& runtime)
    {
        return runtime.getNbElements() - 1 - runtime.getOffset();
    }
    
}; // Navigator<Forward, Frame, jumpSize>
    
    
/** *******************
 * @brief the second implementation of the navigator. This one is used for indexable
 * datatypes. The direction is forwars i.e. is starts at the first element and
 * go to the last one.
 *
 *****************/////    
template<int_fast32_t jumpSize>
struct Navigator<Indexable,
                 hzdr::Direction::Forward, 
                 jumpSize>
{
public:
    
/**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
 */
    template<typename TIndex, typename TContainer, int_fast32_t size=jumpSize>
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
             int_fast32_t size=jumpSize>
    inline
    static
    typename std::enable_if<size == 0>::type
    next(TContainer* ptr, 
         TIndex& index, 
         const TRuntimeVariables& run)
    
    {

        index += run.getJumpsize();
    }
    
    template<typename TRuntimeVariables>
    static
    int_fast32_t 
    inline 
    first(const TRuntimeVariables& run)
    {
        return run.getOffset();
    }
    
}; // Navigator<Backward, Frame, jumpSize>


/** ****************
 * @brief This Navigator can acess all Frames in a Supercell. The direction is
 * forward.
 *****************/

template<typename TFrame,
         int_fast32_t jumpSize>
struct Navigator< hzdr::SuperCell<TFrame>, hzdr::Direction::Forward, jumpSize>
{
    typedef hzdr::SuperCell<TFrame>   SuperCellType;
    typedef TFrame                    FrameType;
    typedef FrameType*                FramePointer;
    
public:
    
    /**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
template<typename TIndex, typename TContainer, int_fast32_t jumps = jumpSize>
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
     * @return true: it is at the end, before the iterations are finished
     *         false: it iterate until it is finished
     */
    template<typename TContainer, typename TRuntime>
    static
    bool
    inline
    next(TContainer*& ptr, const TRuntime& runtimeVariables) 
    {

        for(int_fast32_t i=0; i<runtimeVariables.getJumpsize(); ++i)
        {
            
                       
                        
            if(ptr == nullptr)  {
#pragma omp critical
                std::cout << "I'm" << omp_get_thread_num() << " and " << std::boolalpha << (i + runtimeVariables.getOffset() >= runtimeVariables.getJumpsize()) <<std::endl;
                return i + runtimeVariables.getOffset() >= runtimeVariables.getJumpsize();
            }
            ptr = ptr->nextFrame;
        }
        return false;
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
            for(uint i=0; i < runtimeVariables.getOffset(); ++i)
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
template<typename TFrame, int_fast32_t jumpSize>
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
    template<typename TIndex, typename TContainer, typename TRuntime, int_fast32_t size = jumpSize>
    static
    typename std::enable_if<size != 0, bool>::type
    inline
    next(TContainer* ptr, TIndex& index , const TRuntime& runtimeVariables) 
    {
        boost::ignore_unused(runtimeVariables);
        for(size_t i=0; i<jumpSize; ++i)
        {
            
            if(ptr == nullptr) return i + runtimeVariables.getOffset() < runtimeVariables.getJumpsize();
            ptr = ptr->previousFrame;
        }
        return false;
    }
    
    
    /**
     * @brief runtime implementation of next element implementation. This function
     * is called if the template parameter jumpsize == 0.
     * 
     */
    template<typename TIndex, typename TContainer, typename TRuntime, int_fast32_t size = jumpSize>
    static
    typename std::enable_if<size == 0, bool>::type
    inline
    next(TContainer* ptr, TIndex& index, const TRuntime& run) 
    {

        for(size_t i=0; i<jumpSize; ++i)
        {
            
            if(ptr == nullptr)
            {
#pragma omp critical
                std::cout << "I'm" << omp_get_thread_num() << " and " << std::boolalpha << (i + run.getOffset() < run.getJumpsize()) <<std::endl;
                return i + run.getOffset() < run.getJumpsize();
            }
            ptr = ptr->previousFrame;
        }
        return false;
    }
    
    template<typename TRuntime>
    static 
    FramePointer
  
    first(const SuperCellType* supercell, const TRuntime& run)
    {
        if(supercell != nullptr)
        {
            auto ptr = supercell->lastFrame;
            for(auto i=0; i < run.getOffset(); ++i)
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