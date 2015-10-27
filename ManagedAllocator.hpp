#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>

// This is an allocator to be used in STL containers
// to allocate CUDA Managed memory. Accessible fom host
// and device

template <typename T>
class managed_allocator 
{
public:
    using value_type    = T;
    using size_type     = size_t;
    using pointer       = T*;
    using const_pointer = const T*;
    
    pointer allocate(size_type n) {
        pointer result;
        cudaError_t err = cudaMallocManaged(&result, n * sizeof(T), cudaMemAttachGlobal);
        assert(err == cudaSuccess);
        return result;
    }

    void deallocate(pointer p, size_type) {
        cudaError_t err = cudaFree(p);
        assert(err == cudaSuccess);
    }
};
