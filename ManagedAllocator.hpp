#pragma once

#include <cuda.h>
#include <cassert>

// This is an allocator to be used in STL containers
// to allocate CUDA Managed memory. Accessible fom host
// and device

template <typename T>
class managed_allocator : public std::allocator<T>
{
public:
    using size_type     = size_t;
    using pointer       = T*;
    using const_pointer = const T*;

    pointer allocate(size_type n) {
        pointer tmp;
        cudaError_t err = cudaMallocManaged(&tmp, n * sizeof(T), cudaMemAttachGlobal);
        assert(err == cudaSuccess);
    }

    void deallocate(pointer p, size_type) {
        cudaError_t err = cudaFree(p);
        assert(err == cudaSuccess);
    }
};
