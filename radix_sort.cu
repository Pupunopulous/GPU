#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define BITS_PER_PASS 4  // Number of bits processed per pass
#define RADIX (1 << BITS_PER_PASS)  // Base for each pass (e.g., 16 for 4 bits)

// Helper function to fill the array with random values
void fillArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100; // Random values between 0 and 99
    }
}

// Helper function to check if the array is sorted
int isSorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return 0;
        }
    }
    return 1;
}

// CPU Counting Sort for a specific digit (exp)
void countingSortCPU(int *arr, int n, int exp) {
    int *output = (int *)malloc(n * sizeof(int));
    int count[RADIX] = {0};

    // Count occurrences of each digit
    for (int i = 0; i < n; i++) {
        int digit = (arr[i] / exp) % RADIX;
        count[digit]++;
    }

    // Perform prefix sum to determine positions
    for (int i = 1; i < RADIX; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % RADIX;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }

    // Copy the output array back to arr
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(output);
}

// CPU Radix Sort
void radixSortCPU(int *arr, int n) {
    int maxVal = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > maxVal) {
            maxVal = arr[i];
        }
    }

    // Sort using counting sort for each digit
    for (int exp = 1; maxVal / exp > 0; exp *= RADIX) {
        countingSortCPU(arr, n, exp);
    }
}

// CUDA kernel to perform counting sort on the GPU for each digit
__global__ void countingSort(int *d_arr, int *d_output, int n, int exp) {
    __shared__ int count[RADIX];

    // Initialize count array in shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x < RADIX) count[threadIdx.x] = 0;
    __syncthreads();

    // Count occurrences of each digit
    if (tid < n) {
        int digit = (d_arr[tid] / exp) % RADIX;
        atomicAdd(&count[digit], 1);
    }
    __syncthreads();

    // Perform prefix sum to determine positions
    if (threadIdx.x < RADIX) {
        for (int i = 1; i < RADIX; i++) {
            count[i] += count[i - 1];
        }
    }
    __syncthreads();

    // Sort the elements
    if (tid < n) {
        int digit = (d_arr[tid] / exp) % RADIX;
        int pos = atomicSub(&count[digit], 1) - 1;
        d_output[pos] = d_arr[tid];
    }
}

// Radix Sort function on the GPU
void radixSortGPU(int *d_arr, int n, int threadsPerBlock, int numBlocks) {
    int *d_output;
    cudaMalloc((void **)&d_output, n * sizeof(int));

    int maxVal;
    cudaMemcpy(&maxVal, d_arr, sizeof(int), cudaMemcpyDeviceToHost);

    // Determine the maximum value to know the number of digits
    for (int exp = 1; maxVal / exp > 0; exp *= RADIX) {
        countingSort<<<numBlocks, threadsPerBlock>>>(d_arr, d_output, n, exp);
        cudaMemcpy(d_arr, d_output, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_output);
}

int main(int argc, char *argv[]) {
    // Default values
    int n = 1024;                // Number of elements
    int threadsPerBlock = 256;   // Threads per block
    int numBlocks = 4;           // Number of blocks

    // Parse command-line arguments
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) threadsPerBlock = atoi(argv[2]);
    if (argc >= 4) numBlocks = atoi(argv[3]);

    printf("Array Size: %d, Threads per Block: %d, Number of Blocks: %d\n", n, threadsPerBlock, numBlocks);

    int *arrCPU, *arrGPU, *d_arr;
    clock_t start, end;

    // Allocate memory on host
    arrCPU = (int *)malloc(n * sizeof(int));
    arrGPU = (int *)malloc(n * sizeof(int));

    // Fill array with random values
    fillArray(arrCPU, n);
    memcpy(arrGPU, arrCPU, n * sizeof(int)); // Copy values for GPU

    // CPU Radix Sort
    start = clock();
    radixSortCPU(arrCPU, n);
    end = clock();
    double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU time = %lf seconds\n", cpuTime);

    // Check if CPU sort is correct
    if (!isSorted(arrCPU, n)) {
        printf("CPU sorting failed!\n");
        return 1;
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arrGPU, n * sizeof(int), cudaMemcpyHostToDevice);

    start = clock();
    
    // Call Radix Sort on the GPU
    radixSortGPU(d_arr, n, threadsPerBlock, numBlocks);

    end = clock();
    double gpuTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Copy result back to host
    cudaMemcpy(arrGPU, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Check if GPU sort is correct
    if (!isSorted(arrGPU, n)) {
        printf("GPU sorting failed!\n");
        return 1;
    }

    // Calculate and print speedup
    double speedup = cpuTime / gpuTime;
    printf("GPU time = %lf seconds\n", gpuTime);
    printf("Speedup (CPU/GPU) = %lfx\n", speedup);

    // Free memory
    cudaFree(d_arr);
    free(arrCPU);
    free(arrGPU);

    return 0;
}
