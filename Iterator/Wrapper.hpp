// /**
//  * \struct Wrapper
//  * @author Sebastian Hahn ( t.hahn@hzdr.de )
//  * 
//  * @brief Deserve an interface to get the value of the current iterator position.
//  * 
//  * While the collectiv iteration over all values in the container, it is
//  * possible, that the element is not valid. But after calling ++operator it is
//  * valid. This class has two function:
//  * operator bool: return true if the value is valid and false otherwise
//  * operator* return the value.
//  * The wrapper has one constructor with four parameters:
//  * Wrapper(ElementPtr ptr, 
//             TContainer const * const containerPtr, 
//             TComponent const * const componentenPtr, 
//             const TIndex& pos)
//   The first is a possible pointer to the resulting object. The other three 
//   parameter are internals of the iterator, to decide, whether the pointer is valid.
//   For more Details see DeepIterator.
//  */
// #pragma once
// #include "Iterator/Collective.hpp"
// #include "Definitions/hdinline.hpp"
// #include "Traits/IsIndexable.hpp"
// #include "Traits/NumberElements.hpp"
// 
// namespace hzdr
// {
// 
// 
// 
// 
// template<typename TElement>
// struct Wrapper
// {
//     typedef TElement        ElementType;
//     typedef ElementType*    ElementPtr;
//     
//     template<typename TContainer,
//             typename TComponent,
//             typename TIndex>
//     HDINLINE
//     Wrapper(ElementPtr ptr, 
//             TContainer const * const containerPtr, 
//             TComponent const * const componentenPtr, 
//             const TIndex& pos,
//             typename std::enable_if<traits::IsIndexable<TContainer>::value, int>::type* = 0):
//         ptr(ptr)
//     {
//         typedef traits::NumberElements< TContainer> NbElem;
//         NbElem nbElem;        
//         result = pos >= 0 and pos < nbElem.size(*containerPtr);
//     }
//     
//         template<typename TContainer,
//             typename TComponent,
//             typename TIndex>
//     HDINLINE
//     Wrapper(ElementPtr ptr, 
//             TContainer const * const containerPtr, 
//             TComponent const * const componentenPtr, 
//             const TIndex& pos,
//             typename std::enable_if<not traits::IsIndexable<TContainer>::value, int>::type* = 0
//            ):
//         ptr(ptr),
//         result(ptr != nullptr)
//     {}
//     
//     HDINLINE
//     Wrapper(std::nullptr_t):
//         ptr(nullptr)
//     {}
//     
//     HDINLINE
//     ElementType&
//     operator*()
//     {
//         return *ptr;
//     }
//     
//     HDINLINE
//     explicit
//     operator bool()
//     {
//         return result;
//     }
//     
// protected:
//     ElementPtr ptr;
//     bool result;
//    
//     
// };
// 
// 
// }
