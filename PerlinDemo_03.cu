#include <cmath>
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

    // Typical Perlin noise
    T n = pn.noise(10 * x, 10 * y, 0.8);

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

	// Create a PerlinNoise object with a random permutation vector generated with seed
	unsigned int seed = 237;

    std::vector<int, managed_allocator<int>> permutation = generator(seed);

    const int block_dim_x = 32;
    const int block_dim_y = 16;
    dim3 dimGrid(std::ceil((double) width/(double)block_dim_x),
                 std::ceil((double)height/(double)block_dim_y));
    dim3 dimBlock(block_dim_x, block_dim_y);

	// Visit every pixel of the image and assign a color generated with Perlin noise
	generate_image<float ><<<dimGrid, dimBlock>>>(permutation.data(),
	                                              image.r.data(),
	                                              image.g.data(),
	                                              image.b.data(),
	                                              width, height);
    cudaError_t err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

	// Save the image in a binary PPM file
	image.write("figure_10_P.ppm");

	return 0;
}
