// #pragma once
// #include "Traits/ContainerCategory.hpp"
// #include "Traits/IndexType.hpp"
// 
// namespace hzdr
// {
// 
// namespace traits
// {
// template<
//     typename TContainer,
//     typename TJumpsize = uint_fast32_t,
//     typename SFINAE = void>
// struct PreviousElement;
// 
// template<
//     typename TContainer
//     typename TJumpsize>
// struct PreviousElement<
//     TContainer, 
//     TJumpsize,
//     typename std::enable_if<std::is_same<typename ContainerCategory<TContainer>::type, details::ArrayBased>::value>::type>
// {
//     template<
//         typename TContainer,
//         typename TComponent,
//         typename TIndex>
//     TJumpsize operator()(TContainer, TComponent, TIndex & index, TJumpsize && jumpsize )
//     {
//         index -= jumpsize;
//     }
// };
// 
// template<typename TJumpsize, typename Frame>
// struct PreviousElement<hzdr::Supercell<Frame>, void>
// {
//     template<
//         typename TContainer,
//         typename TComponent,
//         typename TIndex>
//     void operator()(TContainer* containerPtr, TComponent * componentPtr, TIndex &, TJumpsize && jumpsize)
//     {
//         if(componentPtr == nullptr) 
//         {
//             componentPtr = containerPtr->last;
//             --jumpsize;
//         }
//         
//         for(int i=0; i<static_cast<int>(jumpsize); ++i)
//         {
//             if(componentPtr == nullptr) 
//                 break;
//             componentPtr = componentPtr->previous;
//         }
//         
//     }
// };
// 
//     
// } // namespace traits
// } // namespace hzdr
// 
