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
#include "Traits/HasOffset.hpp"
#include <omp.h>
#include "Iterator/RuntimeTuple.hpp"
template<
    typename TElement>
struct DeepIterator;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * @brief Der  ist ein virtueller Container, welcher zusätzliche 
 * Informationen bereithält. Beispielsweiße benötigt ein Frame die Information,
 * wie viele Particle in diesem Frame drin sind. 
 * In diesem Beispiel wollen wir einen  schreiben, welcher nur das 
 * n-te Element betrachtet. Als erstes betrachten wir hierfür den .
 * 
 * Zu betrachtende Beispiele
 * 1. 1d Fall (done
 * 2. 2d Fall
 * 3. nd Fall
 * 4. 1d Fall nur jedes zweites Element
 * 
 * 
 * 
 * 1. Fall
 * Arbeitsreihenfolge:
 * 1. Ich überlege mir, wie ich den  auf den std::vector abbilde
 * 
 * Bedingungen:
 * Damit der Deepcontainer mit DeepForeach arbeitet benötigt er
 * 1. begin() und end()
 * 2. einen definierten iterator datentype (innerhalb der Klasse)
 * 
 * 2. Fall
 * 
 * @tparam TContainer ist der Container Der Daten
 * @tparam TElement Der Rückgabetyp des Iterator. in unserem ersten Beispiel ist
 * es ein int
 */

struct testTrue 
{
    float offset;
};

#define HDINLINE __device__ __forceinline__
#define DINLINE __device__ __forceinline__

struct SyncerCuda
{
    
    DINLINE void sync()
    {
        __syncthreads();
    }
    
    
    
    DINLINE
    void 
    allocSharedMem(int*& sharedMem, int* globalMem)
    {
        
        __shared__ int arr[256];
        sharedMem=arr;
    }
    
    DINLINE
    void 
    loadInSharedMemory(int* sharedMem, int const * const globMem, int const & myId)
    {
        sharedMem[myId] = globMem[myId];
    }
    
    __device__ 
    __forceinline__
    void
    storeInGlobalMemory(int* globMem, int* sharedMem, int const & myId)
    {
        globMem[myId] = sharedMem[myId];
    }
    
};

template<typename TCollective>
class Iterator
{
public:
    typedef TCollective Collective;
public:
    HDINLINE
    Iterator(int *ptr, int myID):
        globalMem(ptr),
        myId(myID)
    {
        collective.allocSharedMem(sharedMem, ptr);
        collective.loadInSharedMemory(sharedMem, globalMem, myId);
        collective.sync();
    }
    
    HDINLINE
    ~Iterator()
    {
        globalMem[myId] = sharedMem[myId];
    }
    
    HDINLINE
    Iterator& 
    operator++()
    {
        globalMem[myId] = sharedMem[myId];
        collective.sync();
        globalMem += 256;
        collective.loadInSharedMemory(sharedMem, globalMem, myId);
        collective.sync();
        return *this;
    }
    HDINLINE
    int& 
    operator*()
    {
        return sharedMem[myId];
    }
    
protected:
    Collective collective;
    int *globalMem;
    int *sharedMem;
    int myId;
};

__global__
void 
myKernel(int* array, const int dim)
{
    const int myId = threadIdx.x; 
    Iterator<SyncerCuda> it(array, myId);
    
    *it += 4;
    ++it; 
    *it += 1;
}

int main(int argc, char **argv) {
/** 1. erstellen eines 2d Arrays auf der GPU. Die zweite Dimension ist dabei 256
 * Elemente groß.
 * 2. Einen Kernel schreiben, der diese Datenstruktur als eingabe parameter nimmt
 * 3. Die Datenstruktur einer Klasse übergeben 
 * 4. Über die datenstruktur iterieren
*/
    const int dim = 2;
    int *array_h;
    int *array_d;
    
    array_h = new int[dim*256];
    
    for(int i=0; i< dim*256; ++i)
    {
        array_h[i] = 1;
    }
    
    gpuErrchk(cudaMalloc(&array_d, sizeof(int) * dim * 256));
    gpuErrchk(cudaMemcpy(array_d, array_h, sizeof(int) * dim * 256, cudaMemcpyHostToDevice));
    myKernel<<<1, 256>>>(array_d, dim);
    
    
    std::cout << "It works" << std::endl;
    
    gpuErrchk(cudaMemcpy(array_h, array_d, sizeof(int) * dim * 256, cudaMemcpyDeviceToHost));
    
    for(int i=0; i<dim; ++i)
    {
        for(int j=0; j<256; ++j)
        {
            std::cout << array_h[j + i*256] << " ";
        }
        std::cout << std::endl;
    }
    

    delete [] array_h;
    return EXIT_SUCCESS;
    
}
