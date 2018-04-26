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
 * @author Sebastian Hahn t.hahn < at > deepiterator.de
 * @brief Each container categorie need to determine if it is a random accessable.
 * A container categorie is random accessable if there is a way to overjump some
 * values easily. Mostly a container is random accessable if the operator [] is
 * overloaded. 
 * @tparam TContainer: type of the container
 * @tparam TContainerCategory: SFIANE type if a categorie is specified
 */
template<
    typename TContainerCategorie, 
    typename TContainerCategory = void
>
struct IsRandomAccessable;


}// namespace traits
}// namespace deepiterator

