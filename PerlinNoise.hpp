#pragma once

#include <vector>

// THIS CLASS IS A TRANSLATION TO C++11 FROM THE REFERENCE
// JAVA IMPLEMENTATION OF THE IMPROVED PERLIN FUNCTION (see http://mrl.nyu.edu/~perlin/noise/)
// THE ORIGINAL JAVA IMPLEMENTATION IS COPYRIGHT 2002 KEN PERLIN

// I ADDED AN EXTRA METHOD THAT GENERATES A NEW PERMUTATION VECTOR (THIS IS NOT PRESENT IN THE ORIGINAL IMPLEMENTATION)

template <typename T>
class PerlinNoise {
public:
	// Initialize with the reference values for the permutation vector
	PerlinNoise(int* vector);
	// Get a noise value, for 2D images z can have any value
	T noise(T x, T y, T z);
private:
	T fade(T t);
	T lerp(T t, T a, T b);
	T grad(int hash, T x, T y, T z);

	// The permutation vector
	int* p;
};

#include "PerlinNoise-impl.hpp"
