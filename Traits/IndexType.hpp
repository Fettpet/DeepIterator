#pragma once


namespace hzdr
{
namespace traits
{

template<typename TContainer>
struct IndexType;
    

template<typename Particle, int_fast32_t size>
struct IndexType< hzdr::Frame<Particle, size> >
{
    typedef int_fast32_t type;
};

template<>
struct IndexType<hzdr::details::UndefinedType>
{
    typedef hzdr::details::UndefinedType type;
};

template<typename Frame>
struct IndexType< hzdr::SuperCell<Frame> >
{
    typedef hzdr::details::UndefinedType type;
};

template<typename TElem, int_fast32_t size>
struct IndexType< hzdr::Particle< TElem, size> >
{
    typedef int_fast32_t type;
};

template<typename Supercell>
struct IndexType< hzdr::SupercellContainer<Supercell> >
{
    typedef int_fast32_t type;
};
}

}
