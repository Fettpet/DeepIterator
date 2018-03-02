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
// traits
#include "deepiterator/traits/Componenttype.hpp"
#include "deepiterator/traits/ContainerCategory.hpp"
#include "deepiterator/traits/IsBidirectional.hpp"
#include "deepiterator/traits/IsRandomAccessable.hpp"
#include "deepiterator/traits/RangeType.hpp"
#include "deepiterator/traits/HasConstantSize.hpp"
#include "deepiterator/traits/NumberElements.hpp"
#include "deepiterator/traits/IndexType.hpp"
// accessor
#include "deepiterator/traits/Accessor/Ahead.hpp"
#include "deepiterator/traits/Accessor/Behind.hpp"
#include "deepiterator/traits/Accessor/Equal.hpp"
#include "deepiterator/traits/Accessor/Get.hpp"

// navigator
#include "deepiterator/traits/Navigator/AfterLastElement.hpp"
#include "deepiterator/traits/Navigator/BeforeFirstElement.hpp"
#include "deepiterator/traits/Navigator/LastElement.hpp"
#include "deepiterator/traits/Navigator/NextElement.hpp"
#include "deepiterator/traits/Navigator/PreviousElement.hpp"
#include "deepiterator/traits/Navigator/FirstElement.hpp"


