/**
 * \struct ContainerCategory
 * @author Sebastian Hahn (t.hahn < at > hzdr) 
 * @brief This trait is used to specify a categorie of a container. Currently are
 * three categories implemented. They are in the folder Iterator/Categorie. If 
 * a container has no categorie, you need to specify the following traits to use
 * the DeepIterator:
 * The first four are needed for the accesor:
 * 1. hzdr::traits::accessor::Ahead
 * 2. hzdr::traits::accessor::Behind
 * 3. hzdr::traits::accessor::Equal
 * 4. hzdr::traits::accessor::Get
 * The next six are needed for the navigator
 * 5. hzdr::traits::navigator::AfterLastElement
 * 6. hzdr::traits::navigator::BeforeFirstElement
 * 7. hzdr::traits::navigator::FirstElement
 * 8. hzdr::traits::navigator::LastElement 
 * 9. hzdr::traits::navigator::NextElement
 * 10. hzdr::traits::navigator::PreviousElement 
 * If a container isnt bidirectional you doesnt need 1, 2, 6, 8, 10.
 * If a container isnt randon accessable you doesnt need 1,2
 */ 

#pragma once

namespace hzdr 
{
namespace traits
{

    
template<typename T>
struct ContainerCategory
{
    typedef T type;
};

}// namespace traits
    
}// namespace hzdr
