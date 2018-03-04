/*
 * Within this test, we check the nd array categorie.
 */

#define BOOST_TEST_MODULE ForwardIterator
#include <boost/test/included/unit_test.hpp>
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Supercell.hpp"
#include "deepiterator/PIC/Frame.hpp"
#include "deepiterator/PIC/SupercellContainer.hpp"
#include "deepiterator/PIC/Particle.hpp"
#include "deepiterator/DeepIterator.hpp"

const int numberElementsInTest = 15;
    
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


/**
 * @brief this is our container to test the ndArray categorie. The container 
 * must must support the following conditions:
 * 1. operator[](TIndex)
 * 2. dim(): This function returns A TIndex with the size in each dimension.
 * 3. 
 */

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


/**
 * Now we define some traits to be able to work with the Container3d.
 */
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
    typename TIndex,
    typename SFIANE
>
struct IndexType<Container3d<
        T, 
        TIndex
    >,
    SFIANE
>
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


/**
 * @brief This is a index type. A Index type for a nd array must support
 * 1. constructor for a range_type
 * 2. operator[]
 * 3. copy constructor
 * 
 */
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

// We test a 3d 
BOOST_AUTO_TEST_CASE(Index3d)
{
    // 3d
    for(int a = 1; a< numberElementsInTest; ++a)
        for(int b=1; b< numberElementsInTest; ++b)
            for(int c=1; c< numberElementsInTest; ++c)
            {
                std::array<int, 3> containerSize{a, b, c};
                

                for(int i=0; i<a; ++i)
                    for(int j=0; j<b; ++j)
                        for(int k=0; k<c; ++k)
                        {
                            std::array<int, 3> idx{i, j, k}; 
                            auto result = hzdr::detail::idxndToInt<3>(idx, containerSize);
                            auto idx2 = hzdr::detail::intToIdxnd<3>(result, containerSize);
                            auto result2 = hzdr::detail::idxndToInt<3>(idx2, containerSize);
                            BOOST_TEST( idx == idx2 );
                            BOOST_TEST( result == result2);
                        }
            }
                
}

BOOST_AUTO_TEST_CASE(Index4d)
{
    // 4d
    for(int a = 1; a< numberElementsInTest; ++a)
        for(int b=1; b< numberElementsInTest; ++b)
            for(int c=1; c< numberElementsInTest; ++c)
                for(int d=1; c< numberElementsInTest; ++c)
                {
                    std::array<int, 4> containerSize{a, b, c, d};
                    

                    for(int i=0; i<a; ++i)
                        for(int j=0; j<b; ++j)
                            for(int k=0; k<c; ++k)
                                for(int l=0; l<d; ++l)
                                {
                                    std::array<int, 4> idx{i, j, k, l}; 
                                    
                                    auto result = hzdr::detail::idxndToInt<4>(idx, containerSize);
                                    auto idx2 = hzdr::detail::intToIdxnd<4>(result, containerSize);
                                    auto result2 = hzdr::detail::idxndToInt<4>(idx2, containerSize);
                                    BOOST_TEST( idx == idx2 );
                                    BOOST_TEST( result == result2);
                                }
                }
}

/**
 * In this test we check the unnested layer
 */
BOOST_AUTO_TEST_CASE(UNNESTED_LAYER)
{
    for(int a=1; a<numberElementsInTest; ++a)
        for(int b=1; b<numberElementsInTest; ++b)
            for(int c=1; c<numberElementsInTest; ++c)
            {
                Container3d<int, TIndex<int> > container(TIndex<int>{a,b,c});
                

                typedef hzdr::SelfValue<uint_fast32_t> Offset;
                typedef hzdr::SelfValue<uint_fast32_t> Jumpsize;
                // 0. We create a concept
                auto && concept = hzdr::makeIteratorPrescription(
                    hzdr::makeAccessor(),
                    hzdr::makeNavigator(
                        Offset(0u),
                        Jumpsize(1u)));
                
                auto && view = hzdr::makeView(
                    container,
                    concept
                );
                // forward
                int sum=0;
                for(auto it=view.begin(); it!=view.end(); ++it)
                {
                    sum +=*it;
                }
                auto n = a * b * c;
                BOOST_TEST(sum == (n * (n+1))/2);
                
                // backward
                sum=0;
                for(auto it=view.rbegin(); it!=view.rend(); --it)
                {
                    sum +=*it;
                }
                BOOST_TEST(sum == (n * (n+1))/2);
            }
}

/**
 * In this test, we check the nd array as first layer, followed by another 
 * layer
 */
BOOST_AUTO_TEST_CASE(Nested_Layer_First)
{
    for(auto a=1; a < numberElementsInTest; ++a)
        for(auto b=1; b< numberElementsInTest; ++b)
            for(auto c=1; c < numberElementsInTest; ++c)
            {
                Container3d<hzdr::Particle<int, 2u>, TIndex<int> > container(TIndex<int>{a,b,c});
                

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
                auto n = a*b*c;
                for(auto it=view.begin(); it!=view.end(); ++it)
                {
                    sum += *it; 
                }
                BOOST_TEST(sum == (n*(n+1)));
                
                sum=0; 
                for(auto it=view.rbegin(); it!=view.rend(); --it)
                {
                    sum += *it; 
                }
                BOOST_TEST(sum == (n*(n+1)));
            }
            
            
    for(auto a=1; a < numberElementsInTest; ++a)
        for(auto b=1; b< numberElementsInTest; ++b)
            for(auto c=1; c < numberElementsInTest; ++c)
            {
                Container3d<hzdr::Particle<int, 2u>, TIndex<int> > container(TIndex<int>{a,b,c});
                

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
                auto n = a*b*c;
                for(auto it=view.begin(); it!=view.end(); it+=2)
                {
                    sum += *it; 
                }
                BOOST_TEST(sum == (n*(n+1)/2));
                
                sum=0; 
                for(auto it=view.rbegin(); it!=view.rend(); it-=2)
                {
                    sum += *it; 
                }
                BOOST_TEST(sum == (n*(n+1)/2));
            }
}


BOOST_AUTO_TEST_CASE(Nested_Layer_Second)
{
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
    auto n = 8;
    
    for(auto it=view.rbegin(); it!=view.rend(); --it)
    {
        std::cout <<*it << std::endl;
       sum += *it;
    }
    BOOST_TEST(sum == (n*(n+1)));
    
    sum=0; 
    
    for(auto it=view.begin(); it!=view.end(); ++it)
    {
       sum += *it;
    }
    BOOST_TEST(sum == (n*(n+1)));
}
