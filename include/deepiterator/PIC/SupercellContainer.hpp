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

#pragma once
#include <array>
#include <vector>
#include <iomanip>
#include <iostream>
#include "deepiterator/iterator/Categorie.hpp"
#include "deepiterator/definitions/hdinline.hpp"
#include "deepiterator/traits/Traits.hpp"
namespace deepiterator
{
template<typename TSupercell>
struct SupercellContainer
{
    
public:
    typedef TSupercell                                          SupercellType;
    typedef SupercellType*                                      SupercellPtr;
    typedef SupercellContainer<TSupercell>                      ThisType;

    
public:
    
    template<typename TIndex>
    HDINLINE
    SupercellContainer(SupercellPtr supercell,
                       const TIndex& nbSupercells):
        nbSupercells(nbSupercells),
        supercells(new SupercellPtr[nbSupercells])
    {

        for(int_fast32_t i=0;i<nbSupercells; ++i)
        {
            supercells[i] = &(supercell[i]);
        }
    }
    
    /**
     * 
     */
    HDINLINE
    SupercellContainer(const int_fast32_t& nbSupercells, 
                       const int_fast32_t& nbFramesInSupercell)
    {
        supercells = new SupercellPtr[nbSupercells];
        for(int_fast32_t i=0; i<nbSupercells; ++i)
        {
            int_fast32_t nbParticleInLastFrame = rand() % nbFramesInSupercell;
            supercells[i] = new TSupercell(nbFramesInSupercell, nbParticleInLastFrame);
        }
    }
    
    HDINLINE
    ~SupercellContainer()
    {
    }
    
    
    
    template<typename TIndex>
    HDINLINE
    SupercellType&
    operator[] (const TIndex& pos)
    {
        return *(supercells[pos]);
    }
    
    template<typename TIndex>
    HDINLINE
    const
    SupercellType&
    operator[] (const TIndex& pos)
    const
    {
        return *(supercells[pos]);
    }
    
    HDINLINE
    uint_fast32_t
    getNbSupercells()
    const
    {
        return nbSupercells;
    }
protected:
    uint_fast32_t nbSupercells;
    SupercellPtr* supercells;
}; // struct SupercellContainer

// traits
namespace traits 
{
template<typename Supercell>
struct Size<deepiterator::SupercellContainer<Supercell> >
{
    typedef deepiterator::SupercellContainer<Supercell> SupercellContainer;
    
    HDINLINE
    int_fast32_t
    operator()( SupercellContainer* element)
    const
    {
        return element->getNbSupercells();
    }
    
} ;


template<
    typename TSupercell>
struct IsBidirectional<SupercellContainer<TSupercell> >
{
    static const bool value = true;
};

template<
    typename TSupercell>
struct IsRandomAccessable<SupercellContainer<TSupercell> >
{
    static const bool value = true;
};
    
template<typename>
struct HasConstantSize;

template<
    typename TSupercell>
struct HasConstantSize<SupercellContainer<TSupercell> >
{
    static const bool value = true;
};

template<typename>
struct ComponentType;

template<
    typename TSupercell>
struct ComponentType<SupercellContainer<TSupercell> >
{
    typedef TSupercell type;
};

template<typename>
struct ContainerCategory;

template<typename TSupercell>
struct ContainerCategory<SupercellContainer<TSupercell> >
{
    typedef deepiterator::container::categorie::ArrayLike type;
};

} // namespace traits


}// namespace deepiterator
