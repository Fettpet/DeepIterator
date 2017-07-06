#pragma once
#include "PIC/Supercell.hpp"
#include <vector>
#include "Definitions/hdinline.hpp"
namespace hzdr
{
template<typename TSupercell>
struct SupercellContainer
{
    
public:
    typedef TSupercell                                          SupercellType;
    typedef SupercellType*                                      SuperCellPtr;
    typedef SupercellContainer<TSupercell>                      ThisType;

    
public:
    
    template<typename TIndex>
    HDINLINE
    SupercellContainer(SuperCellPtr supercell,
                       const TIndex& nbSupercells):
        nbSupercells(nbSupercells),
        supercells(new SuperCellPtr[nbSupercells])
    {

        for(uint_fast32_t i=0;i<nbSupercells; ++i)
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
        supercells = new SuperCellPtr[nbSupercells];
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
    SuperCellPtr* supercells;
}; // struct SupercellContainer
}// namespace hzdr
