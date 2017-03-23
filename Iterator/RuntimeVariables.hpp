#pragma once 


namespace hzdr 
{
struct RuntimeVariables
{
    RuntimeVariables():
        jumpSize(1),
        offset(0),
        elementsRuntime(0)
    {}
    
    RuntimeVariables(const RuntimeVariables& other) = default;
    
    RuntimeVariables(uint_fast32_t jumpSize, uint_fast32_t offset, uint_fast32_t  elementsRuntime):
        jumpSize(jumpSize),
        offset(offset),
        elementsRuntime(elementsRuntime)
        {}
    
    uint_fast32_t jumpSize;
    uint_fast32_t offset;
    uint_fast32_t elementsRuntime;
    
};
}

