#if not defined(HDINLINE)
    #if defined(__CUDA__) || defined(__CUDACC__)
         #define HDINLINE  __device__ __host__ __forceinline__ 
    #else 
         #define HDINLINE  inline
    #endif
#endif
