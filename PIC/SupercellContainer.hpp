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
    typedef typename SupercellType::FrameType                   FrameType;
    typedef typename FrameType::ParticleType                    ParticleType; 
    typedef SupercellContainer<TSupercell>                      ThisType;

    
public:
    
    /**
     * 
     */
    HDINLINE
    SupercellContainer(const int_fast32_t& nbSupercells, 
                       const int_fast32_t& nbFramesInSupercell):
        supercells(nbSupercells)
    {
        
        for(int_fast32_t i=0; i<nbSupercells; ++i)
        {
            int_fast32_t nbParticleInLastFrame = rand() % nbFramesInSupercell;
            supercells[i] = new TSupercell(nbFramesInSupercell, nbParticleInLastFrame);
        }
    }
    
    HDINLINE
    ~SupercellContainer()
    {
        for(uint_fast32_t i=0; i<supercells.size(); ++i)
        {
            delete supercells[i];
        }
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
    
protected:
    std::vector<SuperCellPtr> supercells;
}; // struct SupercellContainer
}// namespace hzdr