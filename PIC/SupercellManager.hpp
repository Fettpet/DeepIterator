#pragma once
#include <cstdio>
#include <cuda.h>

template<typename Supercell, typename Frame>
__global__ 
void
appendFrame(Supercell* supercell, Frame* frame)
{
    frame->previous = nullptr;
    frame->next = nullptr;
    if(supercell->first == nullptr)
    {
        supercell->first = frame;
        supercell->last = frame;
        return;
    }
    Frame* buffer = supercell->first;
    while(buffer->next != nullptr)
    {
        buffer = buffer->next;
    }
    buffer->next = frame;
    frame->previous = buffer;
    supercell->last = frame;
}

template<typename Supercell>
__global__
void 
resetSupercell(Supercell* supercell)
{
    supercell->first =nullptr;
    supercell->last = nullptr;
    
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename Supercell>
struct SupercellHandle
{
    typedef typename Supercell::FrameType Frame;
    
    SupercellHandle(int nbFrames, int nbParticleInLastFrame):
        nbFrames(nbFrames),
        supercellCPU(new Supercell(nbFrames, nbParticleInLastFrame)),
        framePointerCPU( new Frame*[nbFrames]),
        framePointerGPU( new Frame*[nbFrames])
    {   
        for(auto i=0; i<nbFrames; ++i)
        {
            gpuErrchk(cudaMalloc(&framePointerGPU[i], sizeof(Frame)));
        }
        gpuErrchk(cudaMalloc(&supercellGPU, sizeof(Supercell)));
        gpuErrchk(cudaMemcpy(supercellGPU, supercellCPU, sizeof(Supercell), cudaMemcpyHostToDevice));
        
        framePointerCPU[0] = supercellCPU->first;

        for(auto i=1; i<nbFrames; ++i)
        {
            framePointerCPU[i] = framePointerCPU[i-1]->next;
        }
        
        copyHostToDevice();


    }
    
    
    void 
    copyHostToDevice()
    {
        // 1. Alle Frames Kopieren 
        for(auto i=0; i<nbFrames; ++i)
        {
            gpuErrchk(cudaMemcpy(framePointerGPU[i], framePointerCPU[i], sizeof(Frame), cudaMemcpyHostToDevice));
        }
        
        // 2. build the linked list
        resetSupercell<<<1,1>>>(supercellGPU);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaPeekAtLastError() );
        for(auto i=0; i<nbFrames; ++i)
        {
            appendFrame<<<1,1>>>(supercellGPU, framePointerGPU[i]);
            gpuErrchk( cudaDeviceSynchronize() );
            gpuErrchk( cudaPeekAtLastError() );
        }
        
    }
    
    
    void 
    copyDeviceToHost()
    {
                // 1. Alle Frames Kopieren
        for(auto i=0; i<nbFrames; ++i)
        {
            gpuErrchk(cudaMemcpy(framePointerCPU[i], framePointerGPU[i], sizeof(Frame), cudaMemcpyDeviceToHost));
        }
        
                // 2. Linked list wieder erstellen
        for(auto i=0; i<nbFrames-1; ++i)
            framePointerCPU[i]->next = framePointerCPU[i+1];
        
        for(auto i=1; i<nbFrames; ++i)
            framePointerCPU[i]->previous = framePointerCPU[i-1];
    }
    
    int nbFrames;
    
    Supercell *supercellGPU;
    Supercell *supercellCPU;
    Frame **framePointerCPU;
    Frame **framePointerGPU;
    
private:

}; 

