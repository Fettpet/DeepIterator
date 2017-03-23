#pragma once
#include "PIC/Supercell.hpp"
#include <vector>

namespace hzdr
{
template<typename TSupercell>
struct SupercellContainer
{
    
public:
    typedef TSupercell                                          SupercellType;
    typedef SupercellType*                                      SuperCellPtr;
    typedef typename SupercellType::FrameType                   FrameType;
    typedef typename FrameType::ParticleType                    ParticleType; 
    typedef SupercellContainer<TSupercell>                      ThisType;

    
public:
    
    /**
     * 
     */
    SupercellContainer(const uint_fast32_t& nbSupercells, 
                       const uint_fast32_t& nbFramesInSupercell):
        supercells(nbSupercells)
    {
        
        for(uint_fast32_t i=0; i<nbSupercells; ++i)
        {
            uint_fast32_t nbParticleInLastFrame = rand() % nbFramesInSupercell;
            supercells[i] = new TSupercell(nbFramesInSupercell, nbParticleInLastFrame);
        }
    }
    
    ~SupercellContainer()
    {
        for(uint_fast32_t i=0; i<supercells.size(); ++i)
        {
            delete supercells[i];
        }
    }
    
    
    
    template<typename TIndex>
    inline
    SupercellType&
    operator[] (const TIndex& pos)
    {
        return supercells[pos];
    }
    
    template<typename TIndex>
    inline
    const
    SupercellType&
    operator[] (const TIndex& pos)
    const
    {
        return supercells[pos];
    }
    
protected:
    std::vector<SuperCellPtr> supercells;
}; // struct SupercellContainer
}// namespace hzdr