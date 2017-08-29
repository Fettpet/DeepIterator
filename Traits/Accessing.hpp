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
        typename TComponent,
        typename TIndex>
    auto 
    operator() (
        TContainer* container,
        TComponent*,
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
        typename TComponent,
        typename TIndex>
    auto 
    operator() (
        TContainer*, 
        TComponent* component,
        TIndex &&) 
    const
    ->
    typename traits::ComponentType<std::decay<TContainer>::type>::type &    
    {
        return *component;
    }
};
    
}// namespace traits
}// namespace hzdr
