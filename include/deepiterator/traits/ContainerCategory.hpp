/* Copyright 2018 Sebastian Hahn

 * This file is part of DeepIterator.
 *
 * DeepIterator is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DeepIterator is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */



#pragma once

namespace deepiterator 
{
namespace traits
{

/**
 * \struct ContainerCategory
 * @author Sebastian Hahn (t.hahn < at > deepiterator) 
 * @brief This trait is used to specify a categorie of a container. A categorie
 * needs  
 * The first four are needed for the accesor:
 * 1. deepiterator::traits::accessor::Ahead
 * 2. deepiterator::traits::accessor::Behind
 * 3. deepiterator::traits::accessor::Equal
 * 4. deepiterator::traits::accessor::Get
 
 * The next six are needed for the navigator
 * 5. deepiterator::traits::navigator::AfterLastElement
 * 6. deepiterator::traits::navigator::BeforeFirstElement
 * 7. deepiterator::traits::navigator::FirstElement
 * 8. deepiterator::traits::navigator::LastElement 
 * 9. deepiterator::traits::navigator::NextElement
 * 10. deepiterator::traits::navigator::PreviousElement 
 
 * The rest:
 * 11. deepiterator::traits::IndexType
 * 12. deepiterator::traits::IsBidirectional
 * 13. deepiterator::traits::IsRandomAccessable 
 
 * @tparam TContainer: type of the container
 */ 
template<typename TContainer>
struct ContainerCategory
{
    typedef TContainer type;
};

}// namespace traits
    
}// namespace deepiterator
