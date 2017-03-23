#pragma once
#include <type_traits>

namespace hzdr 
{
namespace runtime 
{
enum class offset { enabled, disabled};
enum class jumpsize { enabled, disabled};
enum class nbElements { enabled, disabled};

}// runtime

namespace details 
{
    template<runtime::jumpsize> 
    struct jump;
    
    template<>
    struct jump<runtime::jumpsize::disabled> {};
    
    template<>
    struct jump<runtime::jumpsize::enabled> 
    {
        jump():
            jumpsize(0)
        {}
        
        jump(const jump& other) = default;
        
        jump(const uint_fast32_t& jumpsize):
            jumpsize(jumpsize)
        {}
        
        uint_fast32_t jumpsize;
    };
    
    
    template<runtime::offset> 
    struct off;
    
    template<>
    struct off<runtime::offset::disabled> {};
    
    template<>
    struct off<runtime::offset::enabled> 
    {
        off():
            offset(0)
        {}
        
        off(const off& other) = default;
        
        off(const uint_fast32_t& offset):
            offset(offset)
        {}
            
        uint_fast32_t offset;
    };
    
    
    template<runtime::nbElements> 
    struct nbelem;
    
    template<>
    struct nbelem<runtime::nbElements::disabled> {};
    
    template<>
    struct nbelem<runtime::nbElements::enabled> 
    {
        nbelem(): nbRuntimeElements(0) 
        {}
        
        nbelem(const nbelem& other) = default;
        
        nbelem(const uint_fast32_t& elem):
            nbRuntimeElements(elem)
            {}
            
        uint_fast32_t nbRuntimeElements;
    };
}


template< runtime::offset off, runtime::nbElements nbElem, runtime::jumpsize jump>
struct RuntimeTuple: public details::jump<jump>, public details::off<off>, public details::nbelem<nbElem>
{
    RuntimeTuple(){}
    
    RuntimeTuple(const RuntimeTuple& other) = default;
    
    RuntimeTuple(const uint_fast32_t jumpsize , const uint_fast32_t& nbelem, const uint_fast32_t& offset):
         details::jump<jump>(jumpsize),
         details::off<off>(offset),
         details::nbelem<nbElem>(nbelem)
         
        
    {
       
    }
}; // 
    
typedef RuntimeTuple<runtime::offset::enabled, runtime::nbElements::enabled, runtime::jumpsize::enabled> RuntimeTupleFull;
}// hzdr