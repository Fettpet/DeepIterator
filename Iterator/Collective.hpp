/**
 * @brief The Collectiv policy needs two functions: 
 * 1. bool isMover() specifies the worker which move the current element within
 * the iterator to the next one. 
 * 2. void sync() After the move all worker must be synchronised
 * 
 */

#pragma once
namespace hzdr
{
namespace Collectivity
{
/**
 * @brief The iterator doesn't 
 * */
struct NonCollectiv
{
    constexpr 
    bool 
    isMover()
    {
        return true;
    }
    
    inline 
    void 
    sync()
    {}
    
    
};

}
}
