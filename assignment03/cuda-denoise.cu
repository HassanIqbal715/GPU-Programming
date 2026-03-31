/*![Figure 1: Denoising example (original image by Simpsons, CC BY-SA 3.0, <https://commons.wikimedia.org/w/index.php?curid=8904364>).](denoise.png)

The file [cuda-denoise.c](cuda-denoise.c) contains a serial
implementation of an _image denoising_ algorithm that (to some extent)
can be used to "cleanup" color images. The algorithm replaces the
color of each pixel with the _median_ of the four adjacent pixels plus
itself (_median-of-five_).  The median-of-five algorithm is applied
separately for each color channel (red, green, and blue).

This is particularly useful for removing "hot pixels", i.e., pixels
whose color is way off its intended value, for example due to problems
in the sensor used to acquire the image. However, depending on the
amount of noise, a single pass could be insufficient to remove every
hot pixel; see Figure 1.

The goal of this exercise is to parallelize the denoising algorithm on
the GPU using CUDA. You should launch as many CUDA threads as pixels
in the image, so that each thread is mapped onto a different pixel.

The input image is read from standard input in
[PPM](http://netpbm.sourceforge.net/doc/ppm.html) (Portable Pixmap)
format; the result is written to standard output in the same format.

To compile:

        nvcc cuda-denoise.cu -o cuda-denoise

To execute:

        ./cuda-denoise < input > output

Example:

        ./cuda-denoise < valve-noise.ppm > valve-denoised.ppm

## Files

- [cuda-denoise.cu](cuda-denoise.cu) [hpc.h](hpc.h)
- [valve-noise.ppm](valve-noise.ppm) (sample input)

 ***/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"
typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxcol;  /* Largest color value (Used by the PPM read/write routines) */
    unsigned char *r, *g, *b; /* color channels (arrays of width x height elements each); each value must be less than or equal to maxcol */
} PPM_image;

/**
 * Read a PPM file from file `f`. This function is not very robust; it
 * may fail on perfectly legal PGM images, but works for the provided
 * cat.pgm file.
 */
void read_ppm( FILE *f, PPM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P6") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P6\n")) {
        fprintf(stderr, "FATAL: wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxcol; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxcol));
    if ( img->maxcol > 255 ) {
        fprintf(stderr, "FATAL: maxcol=%d > 255\n", img->maxcol);
        exit(EXIT_FAILURE);
    }
    /* Get the binary data */
    img->r = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->r != NULL);
    img->g = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->g != NULL);
    img->b = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->b != NULL);
    for (int k=0; k<(img->width)*(img->height); k++) {
        nread = fscanf(f, "%c%c%c", img->r + k, img->g + k, img->b + k);
        if (nread != 3) {
            fprintf(stderr, "FATAL: error reading pixel data\n");
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * Write the image `img` to file `f`; is not NULL, use the string
 * `comment` as metadata.
 */
void write_ppm( FILE *f, const PPM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P6\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxcol);
    for (int k=0; k<(img->width)*(img->height); k++) {
        fprintf(f, "%c%c%c", img->r[k], img->g[k], img->b[k]);
    }
}

/**
 * Free all memory used by the structure `img`
 */
void free_ppm( PPM_image* img )
{
    assert(img != NULL);
    free(img->r);
    free(img->g);
    free(img->b);
    img->r = img->g = img->b = NULL; /* not necessary */
    img->width = img->height = img->maxcol = -1;
}

#define BLKDIM 32

/**
 * Swap *a and *b if necessary so that, at the end, *a <= *b
 */
__device__ __host__ 
void compare_and_swap( unsigned char *a, unsigned char *b )
{
    if (*a > *b ) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

__device__ __host__ 
unsigned char *PTR(unsigned char *bmap, int width, int i, int j)
{
    return (bmap + i*width + j);
}

/**
 * Return the median of v[0..4]
 */
__device__ __host__ 
unsigned char median_of_five( unsigned char v[5] )
{
    /* We do a partial sort of v[5] using bubble sort until v[2] is
       correctly placed; this element is the median. (There are better
       ways to compute the median-of-five). */
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v  , v+1 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    return v[2];
}

/**
 * Denoise a single color channel
 */
void denoise( unsigned char *bmap, int width, int height )
{
    unsigned char *out = (unsigned char*)malloc(width*height);
    unsigned char v[5];
    assert(out != NULL);

    memcpy(out, bmap, width*height);
    /* Note that the pixels on the border are left unchanged */
    for (int i=1; i<height - 1; i++) {
        for (int j=1; j<width - 1; j++) {
            v[0] = *PTR(bmap, width, i  , j  );
            v[1] = *PTR(bmap, width, i  , j-1);
            v[2] = *PTR(bmap, width, i  , j+1);
            v[3] = *PTR(bmap, width, i-1, j  );
            v[4] = *PTR(bmap, width, i+1, j  );

            *PTR(out, width, i, j) = median_of_five(v);
        }
    }
    memcpy(bmap, out, width*height);
    free(out);
}

/**
 * GPU kernel to denoise channels
 */
#define CHANNELS 3
#define OUT_TILE_DIM 30
#define IN_TILE_DIM (OUT_TILE_DIM + 2)
#define __CUDACC___ 1

struct ImageChannels {
    unsigned char *ptr[CHANNELS];
};


__global__ void gpu_denoise_kernel(ImageChannels in,
    ImageChannels out, int width, int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * OUT_TILE_DIM + ty - 1;
    int col = blockIdx.x * OUT_TILE_DIM + tx - 1;

    __shared__ unsigned int in_s[CHANNELS][IN_TILE_DIM][IN_TILE_DIM];

    if (row < height && row >= 0 && col < width && col >= 0) {
        // One thread loads all 3 color channels for its assigned pixel
        in_s[0][ty][tx] = (unsigned int)*PTR(in.ptr[0], width, row, col);
        in_s[1][ty][tx] = (unsigned int)*PTR(in.ptr[1], width, row, col);
        in_s[2][ty][tx] = (unsigned int)*PTR(in.ptr[2], width, row, col);
    }

    __syncthreads();

    if (ty >= 1 && ty < IN_TILE_DIM - 1 && tx >= 1 && tx < IN_TILE_DIM - 1) {
        if (row >= 0 && row < height && col >= 0 && col < width) {
            if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
                for(int c = 0; c < CHANNELS; c++) {
                    unsigned char v[5];
                    v[0] = in_s[c][ty][tx];
                    v[1] = in_s[c][ty][tx-1];
                    v[2] = in_s[c][ty][tx+1];
                    v[3] = in_s[c][ty-1][tx];
                    v[4] = in_s[c][ty+1][tx];
                    *PTR(out.ptr[c], width, row, col) = median_of_five(v);
                }
            }
            else {
                *PTR(out.ptr[0], width, row, col) = in_s[0][ty][tx];
                *PTR(out.ptr[1], width, row, col) = in_s[1][ty][tx];
                *PTR(out.ptr[2], width, row, col) = in_s[2][ty][tx];
            }

        }

    }

}

double gpu_denoise(PPM_image *img, int width, int height) 
{
    ImageChannels in;
    ImageChannels out;

    int size = width*height*sizeof(unsigned char);

    for (int i = 0; i < CHANNELS; i++) {
        cudaMalloc((void **) &in.ptr[i], size);
        cudaMalloc((void **) &out.ptr[i], size);
    }

    const double tstart = hpc_gettime();
    cudaMemcpy(in.ptr[0], img->r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(in.ptr[1], img->g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(in.ptr[2], img->b, size, cudaMemcpyHostToDevice);

    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 gridDim(
        ceil((double) width/OUT_TILE_DIM), 
        ceil((double) height/OUT_TILE_DIM), 
        1
    );

    gpu_denoise_kernel<<<gridDim, blockDim>>>(in, out, width, height);

    cudaDeviceSynchronize(); 
    cudaCheckError();

    cudaMemcpy(img->r, out.ptr[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img->g, out.ptr[1], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img->b, out.ptr[2], size, cudaMemcpyDeviceToHost);
    const double elapsed = hpc_gettime() - tstart;

    for (int i = 0; i < CHANNELS; i++) {
        cudaFree(in.ptr[i]);
        cudaFree(out.ptr[i]);
    }

    return elapsed;
}

int main( void )
{
    PPM_image img;
    read_ppm(stdin, &img);

    // const double tstart = hpc_gettime();
    // denoise(img.r, img.width, img.height);
    // denoise(img.g, img.width, img.height);
    // denoise(img.b, img.width, img.height);
    // const double elapsed = hpc_gettime() - tstart;
    // fprintf(stderr, "CPU Execution time %.3f\n", elapsed);

    const double gpuTime = gpu_denoise(&img, img.width, img.height);
    fprintf(stderr, "GPU Execution time %.6f\n", gpuTime);
    
    write_ppm(stdout, &img, "produced by cuda-denoise.cu");
    free_ppm(&img);
    return EXIT_SUCCESS;
}
