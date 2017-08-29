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
#include "Traits/ContainerCategory.hpp"
#include "Traits/RandomAccessable.hpp"

namespace hzdr 
{


template<typename TContainer,
         typename TFirstElement,
         typename TNextElement,
         typename TEndValue,
         typename TIndex>
struct Navigator
{
// some type definitions
public:
    typedef TContainer                                              ContainerType;
    typedef ContainerType*                                          ContainerPtr;
    typedef ContainerType&                                          ContainerRef;
    typedef traits::Componenttype<ContainerType>::type              ComponentType;
    typedef ComponentType                                           ComponentPtr;
    typedef TIndex                                                  IndexType;
    
    typedef TFirstElement                                           FirstElement;
    typedef TNextElement                                            NextElement;
    typedef TEndValue                                               EndValue;
    
    
    
// the core functions
public:
    
// some default constructors
    HDINLINE Navigator() = default;
    HDINLINE Navigator(Navigator const &) = default;
    HDINLINE Navigator(Navigator &&) = default;

    HDINLINE
    Navigator(
            FirstElement && firstElement, 
            NextElement && nextElement,
            EndValue && endValue):
        firstElem(std::forward<FirstElement>(firstElement)),
        nextElem(std::forward<NextElement>(nextElement)),
        endValue(std::forward<EndValue>(endValue))
    {}
    

    HDINLINE
    void 
    next(ContainerPtr containerPtr,  
         ComponentPtr componentPtr,
         IndexType & index)
    const
    {
        nextElem(containerPtr, componentPtr, index);
    }

    HDINLINE
    void 
    first(ContainerPtr containerPtr,
          ComponentPtr componentPtr,
          IndexType & index)
    const
    {
        firstElem(containerPtr, componentPtr, index);
    }
    
    HDINLINE 
    bool 
    isEnd(ContainerPtr containerPtr,
          ComponentPtr componentPtr,
          IndexType & index)
    const 
    {
        return endValue(containerPtr, componentPtr, index);
    }
    
    
    
    
protected:
    FirstElement    firstElem;
    NextElement     nextElem;
    EndValue        endValue;
    
};
    
template<
    typename TContainer,
    typename TFirstElement,
    typename TNextElement,
    typename TEndValue>
HDINLINE 
auto 
makeNavigator(
    TContainer &&,
    TFirstElement && firstElem,
    TNextElement && nextElem,
    TEndValue && endValue)
->
{
    typedef typename std::decay<TContainer>::type                   ContainerType;
    typedef typename std::decay<TFirstElement>::type                FirstType;
    typedef typename std::decay<TNextElement>::type                 NextType;
    typedef typename std::decay<TEndValue>::type                    EndValue;
    typedef typename traits::ContainerCategory<ContainerType>::type ContainerCat;
    typedef typename traits::IndexType<ContainerCat>::type          IndexType;
    
    typedef Navigator<
        ContainerType,
        FirstType,
        NextType,
        EndValue,
        IndexType>                                                  NavigatorType;
        
    return NavigatorType(
        std::forward<TFirstElement>(firstElem),
        std::forward<NextType>(nextElem),
        std::forward<EndValue>(endValue));
        
}

}// namespace hzdr
