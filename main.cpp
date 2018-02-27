
/** \mainpage 
 * # Motivation {#section1}
 * The open source project 
 * <a href="https://github.com/ComputationalRadiationPhysics/picongpu/">
 * PIConGPU </a> has several datastructures. The particles have some attributes 
 * like position and speed. The particles are grouped in frames. A frame has a 
 * maximum number of particles within the frame. The frames are part of supercell.
 * A supercell contains a double linked list of frames. Each frame within a 
 * supercell, except the last one, has the maximum number of particles. A supercell
 * is devided in several cells. The affiliation of a particle to a cell is a 
 * property of the particle.
 * The goal of this project is to write an iterator that can go through the data 
 * structes. It should be possible to go over nested datastructers. For example 
 * over all particles within a supercell. We would use the iterator on GPU and 
 * CPU. So it should be possible to overjump some elements, such that no element 
 * is used more than ones, in a parallel application.
 * 
 * # The DeepIterator {#section2}
 * The DeepIterator class is used to iterator over interleaved data 
 * structures. The simplest example is for an interleaved data structure is 
 * std::vector< std::vector< int > >. The deepiterator iterates over all ints 
 * within the structure. For more details see DeepIterator and View. 
 * \see DeepIterator.hpp \see View.hpp
 * 
 * # Changes in the datastructure {#section3}
 * The number of elements in the last frame was a property of the supercell. This
 * is a problem. To illustrate this, we give an example. We like to iterate 
 * over all particles in all Supercells. Your first attempt was a runtime variable.
 * The user gives the size of the last frame explicitly when the view is created.
 * This also requires a function which decide wheter the last frame is reached.
 * The problem is, if we go over more than on supercell the number of particles 
 * within the last frame changed with each supercell. Our first approach cann't 
 * handle this case. To overcame this, we would need a function that gives the 
 * size of the last frame. But this information is hidden two layers above. This
 * doesnt see like a good design. 
 * 
 * So we decide to change the datastructres of PIConGPU. Each frame has a fixed 
 * size, i.e. how many particle are at most in the frame. We give the frame a 
 * variable numberParticles. This is the number of particles within the frame.
 * The number must be smaller than the maximum number of particles.
 */

#include <iostream>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <typeinfo>
#include <memory>
#include <cstdlib>
#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "View.hpp"
#include <omp.h>
#include "Definitions/hdinline.hpp"


#include "Tests/Cuda/cuda.hpp"
#include "DeepIterator.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "Iterator/Accessor.hpp"
#include "Iterator/Navigator.hpp"
#include "Iterator/Policies.hpp"
#include "Iterator/Prescription.hpp"
#include "Iterator/Categorie/ArrayNDLike.hpp"
#include "Traits/NumberElements.hpp"
#include "Definitions/hdinline.hpp"
#include <boost/timer.hpp>
#include "Iterator/Categorie/ArrayNDLike.hpp"

    
template<
    typename T,
    std::size_t size>
std::ostream& operator<<(
    std::ostream& out, 
    const std::array<T, size>& elem
)
{
    out << "[";
    for(std::size_t i=0; i< size; ++i)
        out << elem[i] << ", ";
    out << "]";
    return out;
}



template<
    typename T,
    typename TIndex
>
struct Container3d
{
    TIndex dimVar;
    std::vector< std::vector< std::vector<T> > > data;
    Container3d():
        dimVar{2,2,2},
        data(
            dimVar[0], 
            std::vector< std::vector<T> >(
                dimVar[1],
                std::vector<T>(
                    dimVar[2],
                    T()
                )
            )
        )
        {
            T counter(0);
            for(int i=0; i<dimVar[0]; ++i)
                for(int j=0; j<dimVar[1]; ++j)
                    for(int k=0; k<dimVar[2]; ++k)
                    {
                        data[i][j][k] = ++counter;
                    }
        }
    
    template<
        typename TIndex_
    >
    explicit
    Container3d(TIndex_ && dim):
        dimVar(std::forward<TIndex_>(dim)),
        data(
            dimVar[0], 
            std::vector< std::vector<T> >(
                dimVar[1],
                std::vector<T>(
                    dimVar[2],
                    T()
                )
            )
        )
        
    {
        T counter(0);
        for(int i=0; i<dimVar[0]; ++i)
            for(int j=0; j<dimVar[1]; ++j)
                for(int k=0; k<dimVar[2]; ++k)
                {
                    data[i][j][k] = ++counter;
                }
        
    }
    
    Container3d&
    operator=(Container3d const &)= default;
    
    
    Container3d&
    operator=(int const &)
    {
        return *this;
    }
    
    TIndex dim()
    {
        return dimVar;
    }
    
    T&
    operator[](TIndex const & idx)
    {
        return data[idx[0]][idx[1]][idx[2]];
    }
    
    
    T const &
    operator[](TIndex const & idx)
    const
    {
        return data[idx[0]][idx[1]][idx[2]];
    }
    
    const 
    TIndex& 
    dim() 
    const
    {
        return dimVar;
    }
};

template<
    typename T,
    typename TIndex
>
std::ostream& operator<<(std::ostream& out, Container3d<T, TIndex> const & con)
{
    for(auto & layer1 : con.data)
    {
        for( auto & layer2 : layer1)
        {
            out << "[";
            for ( auto & elem: layer2)
            {
                out<< elem << ", ";
            }
            out << "],";
        }
        out << std::endl;
    }
    return out;
}

namespace hzdr 
{

// traits
namespace traits 
{

    
template<
    typename T,
    typename TIndex 
>
struct IsBidirectional<Container3d<
    T, 
    TIndex
> >
{
    static const bool value = true;
} ;



template<
    typename T,
    typename TIndex 
>
struct IsRandomAccessable<Container3d<
    T, 
    TIndex
> >
{
    static const bool value = true;
} ;


template<
    typename T,
    typename TIndex 
>
struct HasConstantSize<Container3d<
    T, 
    TIndex
> >
{
    static const bool value = false;
} ;

template<
    typename T,
    typename TIndex 
>
struct ComponentType<Container3d<
    T, 
    TIndex
> >
{
    typedef T type;
} ;


template<
    typename T,
    typename TIndex 
>
struct ContainerCategory<Container3d<
    T, 
    TIndex
> >
{
    typedef hzdr::container::categorie::ArrayNDLike<3> type;
};

template<
    typename T,
    typename TIndex 
>
struct IndexType<Container3d<
    T, 
    TIndex
> >
{
    typedef TIndex type;
};

template<
    typename T,
    typename TIndex,
    typename SFIANE
>
struct RangeType<Container3d<
        T, 
        TIndex
    >,
    SFIANE
>
{
    typedef int type;
};

template<
    typename T,
    typename TIndex 
>
struct NumberElements<Container3d<
    T, 
    TIndex
> >
{
    typedef Container3d<
        T, 
        TIndex
    > Container;
    

    HDINLINE
    int_fast32_t 
    operator()( Container const * const f)
    const
    {
        return f->dim()[0] * f->dim()[1] * f->dim()[2];    
    }
} ;// struct NumberElements

} // namespace traits

}// namespace hzdr


template<
    typename T
>
struct TIndex: public std::array<T, 3>     
{
    using BaseType = std::array<T, 3>;
    TIndex() = default;
    TIndex(const TIndex &) = default;
    TIndex(TIndex &&) = default;
    TIndex(T const & x):
        BaseType{x, 0, 0}
    {}
    
    TIndex& operator=(TIndex const & other) = default;
    TIndex& operator=(TIndex&& other) = default;
    TIndex(
        T const & x,
        T const & y,
        T const & z
    ):
        BaseType{x,y,z}
    {}
};


int main(int , char **) {
    hzdr::Particle<Container3d<int, TIndex<int> >, 2 > container;
    
    std::cout << container;

    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;

    // 0. We create a concept
    auto && concept = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0u),
            Jumpsize(1u)
        ),
        hzdr::makeIteratorPrescription(
            hzdr::makeAccessor(),
            hzdr::makeNavigator(
                Offset(0u),
                Jumpsize(1u)
            )
        )
    );
    
    auto && view = hzdr::makeView(
        container,
        concept
    );
    int sum=0; 
    auto n = 3*4*5;
    
    for(auto it=view.rbegin(); it!=view.rend(); --it)
    {
         std::cout << *it << std::endl; 
    }
//     std::cout << counter << std::endl;
    std::cout << std::boolalpha << sum << " " <<(n * (n+1)/2) << std::endl;

#if 0
    typedef hzdr::Particle<int32_t, 2u> Particle;
    typedef hzdr::Frame<Particle, 10u> Frame;
    typedef hzdr::Supercell<Frame> Supercell;
    typedef hzdr::SelfValue<uint_fast32_t> Offset;
    typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
    typedef hzdr::SelfValue<uint_fast32_t, 256u> Jumpsize_256;
  
    Frame container;    
    std::cout << container << std::endl;
    // forward
    for(int off=0; off<9; ++off)
        for(int jumpsize=1; jumpsize<9; ++jumpsize)
        {
            auto childPrescriptionJump1 = hzdr::makeIteratorPrescription(
                                        hzdr::makeAccessor(),
                                        hzdr::makeNavigator(
                                            Offset(0),
                                            Jumpsize(1)
                                        ),
                                        hzdr::makeIteratorPrescription(
                                            hzdr::makeAccessor(),
                                            hzdr::makeNavigator(
                                                Offset(off),
                                                Jumpsize(jumpsize),
                                                hzdr::Slice<hzdr::slice::IgnoreLastElements, 1>()
                                            )
                                        )
            );
            
            
            auto view = makeView(
                container, 
                childPrescriptionJump1
            );
            int sum=0;
            for(auto it=view.rbegin(); it!=view.rend(); --it)
            {
                sum += *it;
                std::cout << *it << std::endl;
            }
            
            auto idx = 0;
            if(off == 0 and jumpsize == 1)
                idx = 1;
            int checksum=0;
            if(off == 0)
                for(int i=0; i<10; ++i)
                {
                    checksum += container[i][idx];
                }
            std::cout << off << " " << jumpsize << " " << sum << " " << checksum << std::endl;
        }
#endif
}
