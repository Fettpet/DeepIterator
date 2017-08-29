// /**
//  * \struct View
//  * @author Sebastian Hahn (t.hahn@hzdr.de )
//  * 
//  * @brief The View provides functionality for the DeepIterator. The first 
//  * functionality is the construction of the DeepIterator type. The second part
//  * of the functionality is providing the begin and end functions. Last but not 
//  * least the view connects more than one layer.
//  * 
//  * 
//  * We start with the first functionality, the construction of the DeepIterator.
//  * The DeepIterator has several template parameter. For the most of that we had
//  * written some instances. Most of these require template Parameter to work. The 
//  * View build the types for navigator, accesssor and so on.
//  * One design goal is a easy to use interface. From the container of the stl you 
//  * known that all of them has the functions begin and end. The View gives you 
//  * these two functions, I.e. you can use it, like a stl container.
//  * The last functionality is, the view provides a parameter to picture nested
//  * datastructres. This is down with the child template.
//  * @tparam TContainer  This one describes the container, over wich elements you 
//  * would like to iterate. This Templeate need has some Conditions: I. The Trait 
//  * \b IsIndexable need a shape for TContainer. This traits, says wheter 
//  * TContainer is array like (has []-operator overloaded) or list like; II. The
//  * trait \b ComponentType has a specialication for TContainer. This trait gives the type
//  * of the components of TContainer; III. The Funktion \b NeedRuntimeSize<TContainer>
//  * need to be specified. For more details see NeedRuntimeSize.hpp ComponentType.hpp IsIndexable.hpp
//  * @tparam TDirection The direction of the iteration. There are to posibilities
//  * Forward and Backward. For more details see Direction.
//  * @tparam TCollective is used to determine the collective properties of your 
//  * iterator.
//  * @tparam TChild The child is used to describe nested structures.
//  This template has several requirements: 
//     1. it need to spezify an Iterator type. These type need operator++,  operator*,
//         operator=, operator!= and a default constructor.
//     2. it need an WrapperType type
//     3. it need a begin and a end function. The result of the begin function must
//        have the same type as the operator= of the iterator. The result of the 
//        end function must have the same type as the operator!= of the iterator.
//     4. default constructor
//     5. copy constructor
//     6. constructor with childtype and containertype as variables
//     7. refresh(componentType*): for nested datastructures we start to iterate in
//     deeper layers. After the end is reached, in this layers, we need to go to the
//     next element in the current layer. Therefore we had an new component. This 
//     component is given to the child.
//  */
// 
// #pragma once
// #include "DeepIterator.hpp"
// #include "PIC/Frame.hpp"
// #include "PIC/Particle.hpp"
// #include "Iterator/Accessor.hpp"
// #include "Iterator/Navigator.hpp"
// #include "Iterator/Policies.hpp"
// #include "Iterator/Collective.hpp"
// #include "Traits/NumberElements.hpp"
// #include "Definitions/hdinline.hpp"
// #include <type_traits>
// namespace hzdr 
// {
//     
// 
//    
// template<
//     typename TContainer,
//     typename TDirection,
//     typename TOffset,
//     typename TJumpsize,
//     typename TChild = NoChild>
// struct View
// {
// public:
// // Datatypes    
//     typedef TContainer                                                                                              ContainerType;
//     typedef ContainerType*                                                                                          ContainerPtr;
//     typedef typename traits::ComponentType<ContainerType>::type                                                     ComponentType;
//     typedef TChild                                                                                                  ChildType; 
//     typedef Navigator<ContainerType, TDirection, TOffset, TJumpsize>                                                NavigatorType;
//     typedef Accessor<ContainerType>                                                                                 AccessorType;
//     typedef Wrapper< ComponentType>                                                                                 WrapperType;
//     typedef DeepIterator<ContainerType, AccessorType, NavigatorType,
//                                                 WrapperType, ChildType>                                             iterator; 
//     typedef iterator                                                                                                Iterator; 
// 
//     
// public:
//     
//     
// /***************
//  * constructors without childs
//  **************/
//     HDINLINE
//     View():
//         ptr(nullptr)
//         {}
//         /**
//      * @param container The element 
//      */
//         
//     HDINLINE
//     View(ContainerType& container):
//         ptr(&container)
//     {}
//     HDINLINE
//     View(ContainerPtr con):
//         ptr(con)
//     {}
//     
//     HDINLINE
//     View(const View& other) = default;
//     
//     HDINLINE
//     View(View&& other) = default;
//     
//     
//     HDINLINE
//     View(const View& other, ContainerPtr con):
//         offset(other.offset),
//         ptr(con)
//         {}
//     
//     HDINLINE
//     View(const View& other, ContainerPtr con, ChildType& child ):
//         offset(other.offset),
//         ptr(con),
//         childView(child)
//         {}
//     
//     
//     /**
//      * @param container The element 
//      */
//     HDINLINE
//     View(ContainerType& container, ChildType& child):
//         ptr(&container), 
//         childView(child)
//     {}
//     
//     HDINLINE
//     View(ContainerPtr con,ChildType& child):
//         offset(0),
//         ptr(con),
//         childView(child)
//         {}
//         
//     
//     HDINLINE
//     View(ContainerPtr con, ChildType&& child):
//         offset(0),
//         ptr(con),
//         childView(std::move(child))
//     {}
//     
//     
//     HDINLINE
//     View(ContainerType& container, 
//          const TOffset& offset, 
//          const TJumpsize& jumpsize, 
//          ChildType& child):
//         offset(offset),
//         jumpsize(jumpsize),
//         ptr(&container), 
//         childView(child)
//     {}
//     
//         HDINLINE
//     View(ContainerType& container, 
//          const TOffset& offset, 
//          const TJumpsize& jumpsize, 
//          ChildType&& child):
//         offset(offset),
//         jumpsize(jumpsize),
//         ptr(&container), 
//         childView(child)
//     {}
//     
//     HDINLINE
//     View(ContainerPtr con,
//          const TOffset& offset, 
//          const TJumpsize& jumpsize,
//          ChildType& child):
//         offset(offset),
//         jumpsize(jumpsize),
//         ptr(con),
//         childView(child)
//         {}
//         
//     HDINLINE
//     View(ContainerPtr con,
//          const TOffset& offset, 
//          const TJumpsize& jumpsize,
//          ChildType&& child):
//         offset(offset),
//         jumpsize(jumpsize),
//         ptr(con),
//         childView(child)
//         {}
//         
//         
//     HDINLINE    
//     View& operator=(View&& other) = default;
// 
//     
//     HDINLINE
//     View& operator=(const View& other) = default;
//     
//     /**
//      * 1. Iterator with runtime and offset
//      */
//     
//     template< bool test =  std::is_same<ChildType, NoChild>::value>
//     HDINLINE
//     typename std::enable_if<test, Iterator>::type 
//     begin() 
//     const
//     {
//        
//        return Iterator( ptr, AccessorType(), NavigatorType(offset, jumpsize));
//     }
//     
//     
//     template< bool test = not std::is_same<ChildType, NoChild>::value, typename TUnused =void>
//     HDINLINE
//     typename std::enable_if<test, Iterator>::type                                       
//     begin() 
//     const
//     {
//        return Iterator(ptr,childView, offset);
//     }
//     
// 
//     
//     
//     HDINLINE
//     nullptr_t
//     end() 
//     const 
//     {
//             return nullptr;
//     }
// 
//     HDINLINE
//     void
//     setPtr(ContainerPtr _ptr)
//     {
//         ptr = _ptr;
//     }
//     
//     HDINLINE
//     void
//     setPtr(const View& other)
//     {
//         ptr = other.ptr;
//     }
//     
// protected:
//     ContainerPtr ptr;
//     ChildType childView;
//     TOffset offset;
//     TJumpsize jumpsize;
// }; // struct View
// 
// 
// 
// template<typename TContainer,
//          typename TDirection,
//          typename TChild>
// auto make_view(TContainer& container, 
//                TDirection, 
//                const uint_fast32_t& offset, 
//                const uint_fast32_t& jumpsize, 
//                const TChild& child) 
// -> hzdr::View<TContainer, 
//               TDirection,
//               SelfValue<uint_fast32_t>,
//               SelfValue<uint_fast32_t>,
//               TChild>
// {
//     
//     typedef SelfValue<uint_fast32_t> Offset;
//     typedef SelfValue<uint_fast32_t> Jumpsize;
//     return hzdr::View<TContainer, TDirection, Offset, Jumpsize, TChild>(container, Offset(offset), Jumpsize(jumpsize), child);
// }
// 
// 
// template<typename TContainer,
//          typename TDirection>
// auto make_view(TContainer& container, 
//                TDirection, 
//                const uint_fast32_t& offset, 
//                const uint_fast32_t& jumpsize) 
// -> hzdr::View<TContainer,
//               TDirection, 
//               SelfValue<uint_fast32_t>,
//               SelfValue<uint_fast32_t>,
//               hzdr::NoChild>
// {
//     typedef SelfValue<uint_fast32_t> Offset;
//     typedef SelfValue<uint_fast32_t> Jumpsize;
// 
//     return hzdr::View<TContainer, TDirection, Offset, Jumpsize, hzdr::NoChild>(container, Offset(offset), Jumpsize(jumpsize), hzdr::NoChild());
// }
// 
// 
// } // namespace hzdr
