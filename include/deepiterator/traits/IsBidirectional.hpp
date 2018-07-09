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
 * along with DeepIterator.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @author Sebastian Hahn
 * @brief A container is bidirectional it is possible to go to the previous element.
 * The deepiterator has the functions --it and it-- if it is bidirectional.
 * @tparam TContainer: type of the container
 * @tparam TContainerCategory: SFIANE type if a categorie is specified
 */
#pragma once

namespace deepiterator 
{
namespace traits
{
template<
    typename TContainerCategorie, 
    typename SFIANE = void
>
struct IsBidirectional;

}// namespace traits
}// namespace deepiterator


