/**
 * @author Sebastian Hahn
 * @brief A PIConGPU like datastructure. The supercell contains some frames.
 * The frames are in a linked list. Each frame has the pointer nextFrame and 
 * previousFrame. Only the lastFrame frame is not full with particles. The supercell
 * stores the number of particles in the lastFrame frame. Each supercell has two 
 * pointers to frame: firstFrame and lastFrame.
 */
#pragma once
#include <array>
#include <iomanip>
#include <iostream>
#include "deepiterator/iterator/Categorie.hpp"
#include "deepiterator/definitions/hdinline.hpp"
#include "deepiterator/traits/Traits.hpp"

namespace hzdr
{
template<typename TFrame>
struct Supercell
{

    typedef TFrame frame_type;
    typedef TFrame FrameType;
    typedef TFrame ValueType;
    
    HDINLINE 
    Supercell():
        firstFrame(nullptr),
        lastFrame(nullptr)
    {}
    
    HDINLINE 
    Supercell(const Supercell & other)
    {
        firstFrame = other.firstFrame;
        lastFrame = other.lastFrame;
    }
    
    HDINLINE 
    Supercell( Supercell && other)
    {
        firstFrame = other.firstFrame;
        lastFrame = other.lastFrame;
        other.firstFrame = nullptr;
        other.lastFrame = nullptr;
    }
    
    HDINLINE
    ~Supercell() 
    {
        TFrame* cur = firstFrame;
        while(cur != nullptr)
        {
            TFrame* buffer = cur->nextFrame;
            delete cur;
            cur = buffer;
        }
    }
    
    HDINLINE
    Supercell& 
    operator=(const Supercell& other)
    {
        firstFrame = other.firstFrame;
        lastFrame = other.lastFrame;
        return *this;
    }
    
        
    HDINLINE
    Supercell& 
    operator=( Supercell&& other)
    {
        
        firstFrame = other.firstFrame;
        lastFrame = other.lastFrame;
        other.firstFrame = nullptr;
        other.lastFrame = nullptr;
        return *this;
    }
    
    /**
     * @param nbFrames: number of frames within the supercell,
     * @param nbParticle number of particles in the lastFrame frame
     */
    HDINLINE
    Supercell(uint32_t nbFrames, uint32_t nbParticles):
        firstFrame(new TFrame())
    {
        TFrame *curFrame;
        curFrame = firstFrame;
        for(uint32_t i=1; i<nbFrames; ++i)
        {
            curFrame->nextFrame = new TFrame();
            curFrame->nextFrame->previousFrame = curFrame;
            curFrame = curFrame->nextFrame;
        }
        curFrame->nbParticlesInFrame = nbParticles;
        lastFrame = curFrame;
        
        for(uint32_t i=nbParticles; i<TFrame::maxParticlesInFrame; ++i)
        {
            for(uint32_t dim=0; dim < TFrame::Dim; ++dim)
                lastFrame->particles[i].data[dim] = -1;
        }
        
    }
    
    TFrame *firstFrame = nullptr;
    TFrame *lastFrame = nullptr;
 //   uint32_t nbParticlesInLastFrame;
} ; // struct Supercell

// traits
namespace traits 
{
    
template<
    typename TFrame,
    typename SFIANE
>
struct IndexType<
    hzdr::Supercell<TFrame>,
    SFIANE
>
{
    typedef TFrame* type; 
} ;

template<
    typename TFrame, 
    typename SFIANE
>
struct RangeType<
    hzdr::Supercell<TFrame>, 
    SFIANE 
>
{
    typedef int type; 
} ;
    


template<
    typename TFrame>
struct HasConstantSize<Supercell<TFrame> >
{
    static const bool value = false;
} ;


template<
    typename TFrame>
struct ComponentType<Supercell<TFrame> >
{
    typedef TFrame type;
} ;


template<typename TFrame>
struct NumberElements<Supercell<TFrame> >
{  
    typedef Supercell<TFrame> Container;
    
    NumberElements() = default;
    NumberElements(NumberElements const &) = default;
    NumberElements(NumberElements &&) = default;
    
    HDINLINE
    int_fast32_t 
    operator()(Container* container)
    const
    {
        auto result = 0;
        auto tmp = container->firstFrame;
        while(tmp != nullptr)
        {
            tmp =tmp->nextFrame;
            ++result;
        }
        return result;
    }
    
} ; // NumberElements
} // namespace traits


template<typename TFrame>
HDINLINE
std::ostream& operator<<(std::ostream& out, const Supercell<TFrame>& Supercell)
{
    TFrame *curFrame;
    
    curFrame = Supercell.firstFrame;
    
    while(curFrame != nullptr)
    {
        out << *curFrame << std::endl;
        curFrame = curFrame->nextFrame;
    }
    
    return out;
}

namespace traits 
{
template<
    typename TFrame, 
    typename SFIANE
>
struct IsBidirectional<
    hzdr::Supercell<TFrame>, 
    SFIANE
>
{
    static const bool value = true;
} ;    

template<
    typename TFrame,
    typename SFIANE
>
struct IsRandomAccessable<
    hzdr::Supercell<TFrame>, 
    SFIANE
>
{
    static const bool value = true;
} ;
namespace accessor
{

/**
 * @brief get the value of the element, at the iterator positions. \see Get.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TComponent,
    typename TIndex
>
struct Get<
    hzdr::Supercell<TFrame>,
    TComponent, 
    TIndex, 
    SFIANE
>
{
    HDINLINE
    TComponent&
    operator() (hzdr::Supercell<TFrame>*, TIndex& idx)
    {
        return *idx;
    }
} ;    

/**
 * @brief check if both iterators are at the same element. \see Equal.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TComponent,
    typename TIndex
>
struct Equal<
    hzdr::Supercell<TFrame>,
    TComponent, 
    TIndex, 
    SFIANE
>
{
    HDINLINE
    bool
    operator() (
        hzdr::Supercell<TFrame>* con1, 
        TIndex const & idx1, 
        hzdr::Supercell<TFrame>* con2, 
        TIndex const & idx2
    )
    {
        return con1 == con2 && idx1 == idx2;
    }
} ;

 /**
 * @brief Check if the iterator one is ahead the second one. \see Ahead.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TComponent,
    typename TIndex
>
struct Ahead<
    hzdr::Supercell<TFrame>,
    TComponent, 
    TIndex, 
    SFIANE
>
{
    HDINLINE
    bool
    operator() (
        hzdr::Supercell<TFrame>* con1, 
        TIndex const & idx1, 
        hzdr::Supercell<TFrame>* con2, 
        TIndex const & idx2
    )
    {
        if(con1 != con2)
            return false;
        
        TIndex tmp = idx1;
        while(tmp != nullptr)
        {
            tmp = tmp->previousFrame;
            if(tmp == idx2) 
                return true;
           
        }
        return false;
    }
} ;



/**
 * @brief check wheter the iterator 1 is behind the second one. \see Behind.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TComponent,
    typename TIndex
>
struct Behind<
    hzdr::Supercell<TFrame>,
    TComponent, 
    TIndex, 
    SFIANE
>
{
    HDINLINE
    bool
    operator() (
        hzdr::Supercell<TFrame>*, 
        TIndex const & idx1, 
        hzdr::Supercell<TFrame>*, 
        TIndex const & idx2
    )
    {
        TIndex tmp = idx1;
        while(tmp != nullptr)
        {
            tmp = tmp->nextFrame;
            if(tmp == idx2) 
                return true;
            
        }
        return false;
    }
} ;

} // namespace accessor
    
    
namespace navigator
{

/**
 * @brief Implementation to get the first element. \see FirstElement.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TIndex
>
struct FirstElement<
    hzdr::Supercell<TFrame>,
    TIndex, 
    SFIANE
>
{
    HDINLINE
    void
    operator() (
        hzdr::Supercell<TFrame>* container, 
        TIndex & idx
    )
    {
        idx = container->firstFrame;
    }
} ;
/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TIndex,
    typename TRange
>
struct NextElement<
    hzdr::Supercell<TFrame>,
    TIndex,
    TRange,
    SFIANE
>
{

    template<
        typename TContainerSize
    >
    HDINLINE
    TRange
    operator() (
        hzdr::Supercell<TFrame>*, 
        TIndex& idx, 
        TRange const & range,
        TContainerSize &)
    {
        TRange i = 0;
        for(i = 0; i<range; ++i)
        {
            idx = idx->nextFrame;
            if(idx == nullptr)
                break;
        }
        return range - i;
    }
} ;
/**
 * @brief Implementation to check whether the iterator is after the last element.
 * \see AfterLastElement.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TIndex
>
struct AfterLastElement<
    hzdr::Supercell<TFrame>,
    TIndex, 
    SFIANE
>
{
    template<typename TRangeFunction>
    HDINLINE
    bool
    test(
        hzdr::Supercell<TFrame>*, 
        TIndex const & idx, 
        TRangeFunction const &
    )
    const
    {
        return idx == nullptr;
    }
    
    template<typename TRangeFunction>
    HDINLINE
    void
    set(
        hzdr::Supercell<TFrame>*, 
        TIndex & idx,
        TRangeFunction const &
    )
    const
    {
        idx = nullptr;
    }
} ;

/**
 * @brief Set the iterator to the last element. \see LastElement.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TIndex
>
struct LastElement<
    hzdr::Supercell<TFrame>,
    TIndex,
    SFIANE
>
{
    template<typename TSizeFunction>
    HDINLINE
    void
    operator() (
        hzdr::Supercell<TFrame>* containerPtr, 
        TIndex& index, 
        TSizeFunction &&
    )
    {
        index = containerPtr->lastFrame;
        std::cout << "indexPtr " << index << std::endl;
    }
} ;

/**
 * @brief Implementation to get the next element. For futher details \see 
 * NExtElement.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TIndex,
    typename TRange
>
struct PreviousElement<
    hzdr::Supercell<TFrame>,
    TIndex,
    TRange,
    SFIANE
>
{
    
    template<
        typename TContainerSize>
    HDINLINE
    TRange
    operator() (
        hzdr::Supercell<TFrame>*, 
        TIndex& idx, 
        TRange const & jumpsize,
        TContainerSize&)
    {
        TRange i = 0;
        for(i = 0; i<jumpsize; ++i)
        {
            idx = idx->previousFrame;
            if(idx == nullptr)
                return jumpsize - i;
        }

        return jumpsize - i;
    }
} ;

/**
 * @brief Implementation to check whether the iterator is before the fist 
 * element. \see BeforeFirstElement.hpp
 */
template<
    typename TFrame,
    typename SFIANE,
    typename TIndex,
    typename TRange
>
struct BeforeFirstElement<
    hzdr::Supercell<TFrame>,
    TIndex, 
    TRange,
    SFIANE
>
{
    
    template<typename TRangeFunction>
    HDINLINE
    bool
    test(
        hzdr::Supercell<TFrame>*, 
        TIndex const & idx,
        TRange const & offset, 
        TRangeFunction&
    )
    const
    {
        TIndex tmp = idx;
        for(TRange i = static_cast<TRange>(0); i < offset; ++i)
        { 
            if(tmp == nullptr)
                return true;
            tmp = tmp->previousFrame;
        }
        return tmp == nullptr ;
    }
    

    template<typename TRangeFunction>
    HDINLINE
    void
    set(
        hzdr::Supercell<TFrame>*, 
        TIndex & idx,
        TRange const &, 
        TRangeFunction&
    )
    const
    {
        idx = nullptr;
    }
} ;
}
    
} // namespace traits

} // namespace hzdr
