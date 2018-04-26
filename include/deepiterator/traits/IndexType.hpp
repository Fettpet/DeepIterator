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
namespace details
{
struct UndefinedType;
} // namespace details
namespace traits
{

/**
 * @author Sebastian Hahn t.hahn < at > deepiterator.de
 * @brief This trait is used to decide the indextype. The indextype is used to 
 * specify the position to get the current component out of the container.
 * @tparam TContainer: type of the container
 * @tparam TContainerCategory: SFIANE type if a categorie is specified
 */
template<
    typename TContainer, 
    typename TContainerCategory = void>
struct IndexType;


} // namespace traits

} // namespace deepiterator
