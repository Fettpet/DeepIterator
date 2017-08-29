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
namespace details
{
struct UndefinedType;
}
class Indexable;

template<typename TContainer,
         typename TDirection,
         typename TOffset,
         typename TJumpsize,
         typename SFIANE = void>
struct Navigator;
    

template<
        typename TDirection,
        typename TOffset,
        typename TJumpsize>
struct Navigator<details::UndefinedType, TDirection, TOffset, TJumpsize, void>
{
    typedef int                                 ContainerType;
    typedef TDirection                          DirectionType;
    typedef TOffset                             OffsetType;
    typedef TJumpsize                           JumpsizeType;
    
    Navigator(OffsetType&& offset, JumpsizeType&& jumpsize):
        offset(std::forward<OffsetType>(offset)),
        jumpsize(std::forward<JumpsizeType>(jumpsize))
    {    }
    
    int 
    getJumpsize() 
    {
        return jumpsize();
    }
    
    OffsetType offset;
    JumpsizeType jumpsize;
};


/** ****************
 * @brief This one is used for indexable
 * datatypes. It started at the last element and go to the first one.
 *****************/
template<typename TContainer, typename TOffset, typename TJumpsize>
struct Navigator<TContainer,
                 hzdr::Direction::Backward,
                 TOffset,
                 TJumpsize,
                 typename std::enable_if<traits::IsIndexable<TContainer>::value>::type>
{
    typedef TContainer                          ContainerType;
    typedef hzdr::Direction::Backward           DirectionType;
    typedef TOffset                             OffsetType;
    typedef TJumpsize                           JumpsizeType;
    
public:
    
    
    HDINLINE
    Navigator() = default;
    
    
    HDINLINE
    Navigator(TOffset const & offset, 
              TJumpsize const & jumpsize):
        offset(offset),
        jumpsize(jumpsize)
    {}
            
    
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
        index -= jumpsize();
    }
    
    template<typename TComponent,
             typename TIndex>
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index)
    const
    {
        typedef traits::NumberElements< TContainer> NbElem;
        NbElem nbElem;   
        conPtrOut = conPtrIn;
        index = nbElem.size(*conPtrOut) - 1 - offset();
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
        return static_cast<int_fast32_t>(index) < 0;
    }
    
//protected:
    TOffset offset;
    TJumpsize jumpsize;
}; // Navigator<Forward, Frame, jumpSize>
    
    
/** *******************
 * @brief This one is used for indexable
 * datatypes. The direction is forwars i.e. is starts at the first element and
 * go to the last one.
 *****************/////    
template<typename TContainer, 
         typename TOffset, 
         typename TJumpsize>
struct Navigator<TContainer,
                 hzdr::Direction::Forward,
                 TOffset,
                 TJumpsize,
                 typename std::enable_if<traits::IsIndexable<TContainer>::value>::type > 
{
    typedef void                                ContainerType;
    typedef hzdr::Direction::Forward            DirectionType;
    typedef TOffset                             OffsetType;
    typedef TJumpsize                           JumpsizeType;
public:
    
    HDINLINE 
    Navigator() = default;
    
    HDINLINE
    Navigator(TOffset const & offset, 
              TJumpsize const & jumpsize):
        offset(offset),
        jumpsize(jumpsize)
    {
    }
    
    HDINLINE 
    Navigator(Navigator const & other):
        offset(other.offset),
        jumpsize(other.jumpsize)
    {
    }
    
    HDINLINE 
    Navigator(Navigator&& navi):
        offset(std::forward<TOffset>(navi.offset)),
        jumpsize(std::forward<TJumpsize>(navi.jumpsize))
    {
    };
    
    HDINLINE
    Navigator(TOffset && offset, 
              TJumpsize && jumpsize):
        offset(std::forward<TOffset>(offset)),
        jumpsize(jumpsize)
    {
    }
    
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
        index += jumpsize();
    }
    
    int 
    getJumpsize() 
    {
        return jumpsize();
    }
    
    template<typename TIndex,
            typename TComponent>
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index)
    const
    {

        conPtrOut = conPtrIn;
        index = offset();
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
        return static_cast<int_fast32_t>(index) >= static_cast<int_fast32_t>(nbElem.size(*containerPtr));
    }
    
//protected:
    TOffset offset;
    TJumpsize jumpsize;
}; // Navigator<Backward, Frame, jumpSize>


/** ****************
 * @brief This Navigator can acess all Frames in a Supercell. The direction is
 * forward.
 *****************/

template<typename TFrame,
         typename TOffset,
         typename TJumpsize>
struct Navigator< 
    hzdr::SuperCell<TFrame>, 
    hzdr::Direction::Forward,
    TOffset,
    TJumpsize,
    void> 
{
    
    typedef hzdr::Direction::Forward            DirectionType;
    typedef TOffset                             OffsetType;
    typedef TJumpsize                           JumpsizeType;
    typedef hzdr::SuperCell<TFrame>             ContainerType;
    typedef hzdr::SuperCell<TFrame>             SuperCellType;
    typedef TFrame                              FrameType;
    typedef FrameType*                          FramePointer;
    
public:
    
    
    HDINLINE 
    Navigator() = default;
    
    HDINLINE
    Navigator(TOffset const & offset, 
              TJumpsize const & jumpsize):
        offset(offset),
        jumpsize(jumpsize)
    {}
    
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

        for(uint_fast32_t i=0; i<jumpsize(); ++i)
        {
                             
            if(elem == nullptr) 
            {
                ptr = nullptr;
                break;
            }
            elem = elem->nextFrame;
        }
    }

    
    template<typename TComponent,
             typename TIndex,
             typename TContainer>
    
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index)
    const
    {
         
        if(conPtrIn != nullptr)
        {
            compontPtr = conPtrIn->firstFrame;
            for(uint i=0; i < offset(); ++i)
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
    
//protected:
    TOffset offset;
    TJumpsize jumpsize;
}; // Navigator<Forward, Frame, jumpSize>
    
    
    
/**
 * @brief this implementation use supercells. The direction is backward. 
 */    
template<typename TFrame,
         typename TOffset,
         typename TJumpsize>
struct Navigator< hzdr::SuperCell<TFrame>, 
                  hzdr::Direction::Backward,
                  TOffset,
                  TJumpsize,
                void>
{
    typedef hzdr::Direction::Backward           DirectionType;
    typedef TOffset                             OffsetType;
    typedef TJumpsize                           JumpsizeType;
    typedef hzdr::SuperCell<TFrame>             ContainerType;
    typedef hzdr::SuperCell<TFrame>             SuperCellType;
    typedef TFrame                              FrameType;
    typedef FrameType*                          FramePointer;
public:

    
    HDINLINE 
    Navigator() = default;
    
    HDINLINE
    Navigator(TOffset const & offset, 
              TJumpsize const & jumpsize):
        offset(offset),
        jumpsize(jumpsize)
    {}
    
    /**
 * @brief compiletime implementation of next element implementation. This function
 * is called if the template parameter jumpsize != 0.
 */
    template<typename TIndex, 
             typename TContainer>
    HDINLINE
    
    void 
    next(TContainer*& ptr, 
         typename traits::ComponentType<TContainer>::type*& elem,
         TIndex& index)
    const
    {

        for(size_t i=0; i<jumpsize(); ++i)
        {
            
            if(elem == nullptr) 
            {
                ptr = nullptr;
                break;
            }
            elem = elem->previousFrame;
        }

    }
    
    template<typename TIndex,
             typename TComponent,
             typename TContainer>
    
    void 
    HDINLINE 
    first(TContainer* conPtrIn,
          TContainer*& conPtrOut, 
          TComponent*& compontPtr,
          TIndex& index)
    const
    {
       
        if(conPtrIn != nullptr)
        {

            compontPtr = conPtrIn->lastFrame;
            for(auto i=0; i < offset(); ++i)
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
    
protected:
    TOffset offset;
    TJumpsize jumpsize;
}; // Navigator<Forward, Frame, jumpSize>



namespace details
{
struct UndefinedType;
    
template<
    typename TContainer,
    typename TDirection,
    typename TOffset,
    typename TJumpsize
>
HDINLINE 
auto 
makeNavigator(hzdr::Navigator<details::UndefinedType, TDirection, TOffset, TJumpsize>&& other)
-> hzdr::Navigator<TContainer, TDirection, TOffset, TJumpsize>
{
    typedef hzdr::Navigator<TContainer, TDirection, TOffset, TJumpsize> Navi;
    return Navi(std::forward<TOffset>(other.offset), 
                std::forward<TJumpsize>(other.jumpsize));
}

}

template<
    typename TContainer,
    typename TDirection,
    typename TOffset,
    typename TJumpsize
>
HDINLINE 
auto 
makeNavigator(TContainer&&,
               TDirection&&, 
               TOffset&& offset, 
               TJumpsize&& jumpsize)
-> hzdr::Navigator<typename std::remove_reference<TContainer>::type, TDirection, TOffset, TJumpsize>
{
    typedef typename std::remove_reference<TContainer>::type ContainerType;
    typedef hzdr::Navigator<ContainerType, TDirection, TOffset, TJumpsize> Navi;
    return Navi(std::forward<TOffset>(offset), 
                std::forward<TJumpsize>(jumpsize));
}

template<
    typename TDirection,
    typename TOffset,
    typename TJumpsize
>
HDINLINE 
auto 
makeNavigator(TDirection&&, 
               TOffset&& offset, 
               TJumpsize&& jumpsize)
-> hzdr::Navigator<details::UndefinedType, TDirection, TOffset, TJumpsize>
{
    typedef hzdr::Navigator<details::UndefinedType, TDirection, TOffset, TJumpsize> Navi;
    return Navi(std::forward<TOffset>(offset), 
                std::forward<TJumpsize>(jumpsize));
}


}// namespace hzdr
