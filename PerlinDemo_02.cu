#include <cmath>
#include <cassert>
#include "ppm.hpp"
#include "PerlinNoise.hpp"
#include "ManagedAllocator.hpp"

template <typename T>
__global__
void generate_image(const int*     __restrict__ perm, 
                    unsigned char* __restrict__ r,
                    unsigned char* __restrict__ g,
                    unsigned char* __restrict__ b,
                    unsigned int width, unsigned int height) {
    const unsigned int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int kk = i * width + j;

    if (i >= height or j >= width) return;

    T x = (T)j/((T)width);
    T y = (T)i/((T)height);

    // Create a PerlinNoise object with the reference permutation vector
    PerlinNoise<T> pn(perm);

    // Wood like structure
    T n = 20 * pn.noise(x, y, 0.8);
    n -= floor(n);

    // Map the values to the [0, 255] interval, for simplicity we use 
    // tones of grey
    r[kk] = floor(255 * n);
    g[kk] = floor(255 * n);
    b[kk] = floor(255 * n);
}

int main() {
	// Define the size of the image
    unsigned int width = 4096, height = 2160; // 4K

	// Create an empty PPM image
	ppm image(width, height);

    std::vector<int, managed_allocator<int>> permutation =
      { 151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
		8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
		35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,
		134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
		55,46,245,40,244,102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
		18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
		250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
		189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 
		43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,
		97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,
		107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };

    // Duplicate the permutation vector
	permutation.insert(permutation.end(), permutation.begin(), permutation.end());

    const int block_dim_x = 16;
    const int block_dim_y = 16;
    dim3 dimGrid(std::ceil((double) width/(double)block_dim_x),
                 std::ceil((double)height/(double)block_dim_y));
    dim3 dimBlock(block_dim_x, block_dim_y);

	// Visit every pixel of the image and assign a color generated with Perlin noise
	generate_image<float><<<dimGrid, dimBlock>>>(permutation.data(),
	                                             image.r.data(),
	                                             image.g.data(),
	                                             image.b.data(),
	                                             width, height);
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

	// Save the image in a binary PPM file
	image.write("figure_9_R.ppm");

	return 0;
}
