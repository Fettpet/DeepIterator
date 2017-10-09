#pragma once
#include <cstdio>
#include <cuda.h>
#if not defined(gpuErrchk)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

template<typename Supercell, typename Frame>
__global__ 
void
appendFrame(Supercell* supercell, Frame* frame, int i)
{
    frame->previous = nullptr;
    frame->next = nullptr;
    if(supercell[i].first == nullptr)
    {
        supercell[i].first = frame;
        supercell[i].last = frame;
        return;
    }
    Frame* buffer = supercell[i].first;
    while(buffer->next != nullptr)
    {
        buffer = buffer->next;
    }
    buffer->next = frame;
    frame->previous = buffer;
    supercell[i].last = frame;
}

template<typename Supercell>
__global__
void 
resetSupercell(Supercell* supercell, int id)
{
    supercell[id].first =nullptr;
    supercell[id].last = nullptr;
}



template<typename Supercell>
struct SupercellContainerManager
{
    typedef typename Supercell::FrameType Frame;
    
    SupercellContainerManager(int nbSupercells, std::vector<int> nbFramesSupercell, std::vector<int> nbParticlesInLastFrame):
        nbFramesSupercell(nbFramesSupercell),
        supercellCPU(new Supercell[nbSupercells]),
        framePointerCPU( new Frame**[nbSupercells]),
        framePointerGPU( new Frame**[nbSupercells])
    {   
        
        
        for(auto i=0; i<nbSupercells; ++i)
        {
            supercellCPU[i] = Supercell(nbFramesSupercell[i], nbParticlesInLastFrame[i]);// << std::endl;;
            framePointerCPU[i] = new Frame*[nbFramesSupercell[i]];
            framePointerGPU[i] = new Frame*[nbFramesSupercell[i]];

            
            
            for(auto j=0; j< nbFramesSupercell[i]; ++j)
            {
                framePointerCPU[i][j] = new Frame();
                gpuErrchk(cudaMalloc(&framePointerGPU[i][j], sizeof(Frame)));
            }
            framePointerCPU[i][0] = supercellCPU[i].first;
            
            for(auto j=1; j<nbFramesSupercell[i]; ++j)
            {
                framePointerCPU[i][j] = framePointerCPU[i][j-1]->next;
            }
        }
        gpuErrchk(cudaMalloc(&supercellGPU, nbSupercells * sizeof(Supercell)));
        gpuErrchk(cudaMemcpy(supercellGPU, supercellCPU, nbSupercells * sizeof(Supercell), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(supercellCPU, supercellGPU, nbFramesSupercell.size() * sizeof(Supercell), cudaMemcpyDeviceToHost));
        
        
        
        copyHostToDevice();


    }
    
    
    void 
    copyHostToDevice()
    {
        
        // 1. Alle Frames Kopieren 
        for(uint i=0; i<nbFramesSupercell.size(); ++i)
        {
            resetSupercell<<<1,1>>>(supercellGPU, i);
            gpuErrchk( cudaDeviceSynchronize() );
            gpuErrchk( cudaPeekAtLastError() );
            for(auto j=0; j<nbFramesSupercell[i]; ++j)
            {

                gpuErrchk(cudaMemcpy(framePointerGPU[i][j], framePointerCPU[i][j], sizeof(Frame), cudaMemcpyHostToDevice));
                appendFrame<<<1,1>>>(supercellGPU, framePointerGPU[i][j], i);
                gpuErrchk( cudaDeviceSynchronize() );
                gpuErrchk( cudaPeekAtLastError() );
            }
        }
        
    }
    
    
    void 
    copyDeviceToHost()
    {
//         gpuErrchk(cudaMemcpy(supercellCPU, supercellGPU, nbFramesSupercell.size() * sizeof(Supercell), cudaMemcpyDeviceToHost));
                // 1. Alle Frames Kopieren
        for(uint i=0; i<nbFramesSupercell.size(); ++i)
        {
            supercellCPU[i].first = framePointerCPU[i][0];
            for(auto j=0; j<nbFramesSupercell[i]; ++j)
            {
                gpuErrchk(cudaMemcpy(framePointerCPU[i][j], framePointerGPU[i][j], sizeof(Frame), cudaMemcpyDeviceToHost));
            }
            
                    // 2. Linked list wieder erstellen
            for(auto j=0; j<nbFramesSupercell[i]-1; ++j)
                framePointerCPU[i][j]->next = framePointerCPU[i][j+1];
            
            for(auto j=1; j<nbFramesSupercell[i]; ++j)
                framePointerCPU[i][j]->previous = framePointerCPU[i][j-1];
        }
    }
    
    std::vector<int> nbFramesSupercell;
    
    Supercell *supercellGPU;
    Supercell *supercellCPU;
    Frame ***framePointerCPU;
    Frame ***framePointerGPU;
    
private:

}; 
