#pragma once
/**
 * 
 * 
 */

namespace hzdr
{

namespace traits
{

template<typename TType>
struct Accessing;

template<>
struct Accessing<hzdr::traits::details::ArrayBased>
{
    Accessing() = default;
    Accessing(const Accessing &) = default;
    Accessing( Accessing&&) = default;
    
    template<
        typename TContainer,
        typename TIndex>
    auto 
    operator() (
        TContainer* container, 
        TIndex && index) 
    const
    ->
    typename traits::ComponentType<std::decay<TContainer>::type>::type &    
    {
        return (*container)[index];
    }
};

template<>
struct Accessing<hzdr::details::ListBased>
{
    Accessing() = default;
    Accessing(const Accessing &) = default;
    Accessing(Accessing&&) = default;
    
    template<
        typename TContainer,
        typename TIndex>
    auto 
    operator() (
        TContainer* container, 
        TIndex && index) 
    const
    ->
    typename traits::ComponentType<std::decay<TContainer>::type>::type &    
    {
        if(std::is_pointer<TIndex>::value)
            return *index;
        else 
            return index;
    }
};
    
}// namespace traits
}// namespace hzdr
