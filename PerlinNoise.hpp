#pragma once

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

// THIS CLASS IS A TRANSLATION TO C++11 FROM THE REFERENCE
// JAVA IMPLEMENTATION OF THE IMPROVED PERLIN FUNCTION (see http://mrl.nyu.edu/~perlin/noise/)
// THE ORIGINAL JAVA IMPLEMENTATION IS COPYRIGHT 2002 KEN PERLIN

// I ADDED AN EXTRA METHOD THAT GENERATES A NEW PERMUTATION VECTOR (THIS IS NOT PRESENT IN THE ORIGINAL IMPLEMENTATION)

template <typename T>
class PerlinNoise {
public:
	// Initialize with the reference values for the permutation vector
	__forceinline__ __host__ __device__ PerlinNoise(int* vector);
	// Get a noise value, for 2D images z can have any value
	__forceinline__ __host__ __device__ T noise(T x, T y, T z);
private:
	__forceinline__ __host__ __device__ T fade(T t);
	__forceinline__ __host__ __device__ T lerp(T t, T a, T b);
	__forceinline__ __host__ __device__ T grad(int hash, T x, T y, T z);

	// The permutation vector
	int* p;
};

#include "PerlinNoise-impl.hpp"
