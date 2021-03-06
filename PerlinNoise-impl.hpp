#include "PerlinNoise.hpp"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "ManagedAllocator.hpp"

// THIS IS A DIRECT TRANSLATION TO C++11 FROM THE REFERENCE
// JAVA IMPLEMENTATION OF THE IMPROVED PERLIN FUNCTION (see http://mrl.nyu.edu/~perlin/noise/)
// THE ORIGINAL JAVA IMPLEMENTATION IS COPYRIGHT 2002 KEN PERLIN

// I ADDED AN EXTRA METHOD THAT GENERATES A NEW PERMUTATION VECTOR (THIS IS NOT PRESENT IN THE ORIGINAL IMPLEMENTATION)

// Initialize with the reference values for the permutation vector
template <typename T>
__forceinline__ __host__ __device__ 
PerlinNoise<T>::PerlinNoise(const int* vector) // Initialize the permutation vector with the reference values
  : p {vector}
{}

template <typename T>
T PerlinNoise<T>::noise(T x, T y, T z) const {
	// Find the unit cube that contains the point
	int X = static_cast<int>(floor(x)) & 255;
	int Y = static_cast<int>(floor(y)) & 255;
	int Z = static_cast<int>(floor(z)) & 255;

	// Find relative x, y,z of point in cube
	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	// Compute fade curves for each of x, y, z
	T u = fade(x);
	T v = fade(y);
	T w = fade(z);

	// Hash coordinates of the 8 cube corners
	int A  = p[X    ] + Y;
	int AA = p[A    ] + Z;
	int AB = p[A + 1] + Z;
	int B  = p[X + 1] + Y;
	int BA = p[B    ] + Z;
	int BB = p[B + 1] + Z;

	// Add blended results from 8 corners of cube
	T res = lerp(w, 
	             lerp(v, lerp(u, grad(p[AA  ], x, y, z  ), grad(p[BA  ], x-1, y, z  )), lerp(u, grad(p[AB  ], x, y-1, z  ), grad(p[BB  ], x-1, y-1, z  ))),	
	             lerp(v, lerp(u, grad(p[AA+1], x, y, z-1), grad(p[BA+1], x-1, y, z-1)), lerp(u, grad(p[AB+1], x, y-1, z-1),	grad(p[BB+1], x-1, y-1, z-1)))
	);
	return (res + 1.0)/2.0;
}

template <typename T>
__forceinline__ __host__ __device__ 
T PerlinNoise<T>::fade(T t) const { 
	return t * t * t * (t * (t * 6 - 15) + 10);
}

template <typename T>
__forceinline__ __host__ __device__ 
T PerlinNoise<T>::lerp(T t, T a, T b) const { 
	return a + t * (b - a); 
}

template <typename T>
__forceinline__ __host__ __device__ 
T PerlinNoise<T>::grad(int hash, T x, T y, T z) const {
	int h = hash & 15;
	// Convert lower 4 bits of hash inot 12 gradient directions
	T u = h < 8 ? x : y;
	T v = h < 4 ? y : h == 12 || h == 14 ? x : z;
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// Out of class funcitons:

// Generate a new permutation vector based on the value of seed
std::vector<int, managed_allocator<int>> generator(unsigned int seed)
{
    std::vector<int, managed_allocator<int>> p(256);

	// Fill p with values from 0 to 255
	std::iota(p.begin(), p.end(), 0);

	// Initialize a random engine with seed
	std::default_random_engine engine(seed);

	// Suffle  using the above random engine
	std::shuffle(p.begin(), p.end(), engine);

	// Duplicate the permutation vector
	p.insert(p.end(), p.begin(), p.end());

	return p;
}
