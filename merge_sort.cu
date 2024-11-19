#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// Function to print an array
void printArray(int* arr, int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// Helper function to fill the array with random values
void fillArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100; // Random values between 0 and 99
    }
}

// Helper function to check if the array is sorted
int isSorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}

// Sequential merge function
void sequentialMerge(int* input, int* output, int start, int mid, int end) {
    int i = start, j = mid, k = start;
    while (i < mid && j < end) {
        if (input[i] <= input[j]) {
            output[k++] = input[i++];
        } else {
            output[k++] = input[j++];
        }
    }
    while (i < mid) output[k++] = input[i++];
    while (j < end) output[k++] = input[j++];
}

// Sequential merge sort
void sequentialMergeSort(int* data, int size) {
    int* temp = (int*)malloc(size * sizeof(int));
    for (int width = 1; width < size; width *= 2) {
        for (int i = 0; i < size; i += 2 * width) {
            int start = i;
            int mid = (i + width < size) ? (i + width) : size;
            int end = (i + 2 * width < size) ? (i + 2 * width) : size;
            sequentialMerge(data, temp, start, mid, end);
        }
        // Copy temp array back to data
        for (int i = 0; i < size; i++) {
            data[i] = temp[i];
        }
    }
    free(temp);
}

// GPU kernel to perform merging of two sorted subarrays
__global__ void mergeKernel(int* input, int* output, long long width, long long size) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = tid * width * 2;
    if (start >= size) return;

    long long mid = min(start + width, size);
    long long end = min(start + 2 * width, size);

    long long i = start, j = mid, k = start;
    while (i < mid && j < end) {
        if (input[i] <= input[j]) {
            output[k++] = input[i++];
        } else {
            output[k++] = input[j++];
        }
    }
    while (i < mid) output[k++] = input[i++];
    while (j < end) output[k++] = input[j++];
}

// GPU-based merge sort
void gpuMergeSort(int* d_data, long long size, int threadsPerBlock) {
    int* d_temp;
    cudaMalloc(&d_temp, size * sizeof(int));

    for (long long width = 1; width < size; width *= 2) {
        long long numMerges = (size + (width * 2) - 1) / (width * 2);
        long long gridSize = (numMerges + threadsPerBlock - 1) / threadsPerBlock;
        printf("width: %lld, numMerges: %lld, gridSize: %lld\n", width, numMerges, gridSize);

        mergeKernel<<<gridSize, threadsPerBlock>>>(d_data, d_temp, width, size);
        cudaMemcpy(d_data, d_temp, size * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_temp);
}


int main(int argc, char *argv[]) {
    int n = 1024 * 1024;         // Number of elements (default: 1M elements)
    int threadsPerBlock = 1024;  // Threads per block

    // Parse command-line arguments
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) threadsPerBlock = atoi(argv[2]);

    // Allocate and fill host array
    int* h_data = (int*)malloc(n * sizeof(int));
    int* h_data_seq = (int*)malloc(n * sizeof(int));
    fillArray(h_data, n);
    memcpy(h_data_seq, h_data, n * sizeof(int));

    // Measure time for sequential merge sort
    clock_t start_seq = clock();
    sequentialMergeSort(h_data_seq, n);
    clock_t end_seq = clock();
    double time_seq = ((double)(end_seq - start_seq)) / CLOCKS_PER_SEC;

    if (isSorted(h_data_seq, n)) {
        printf("Sequential: Array is sorted correctly.\n");
    } else {
        printf("Sequential: Array is NOT sorted correctly.\n");
    }

    // Print the sequentially sorted array
    // printf("Sequentially sorted array:\n");
    // printArray(h_data_seq, n);

    // GPU merge sort
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    // Measure time for GPU merge sort
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);

    gpuMergeSort(d_data, n, threadsPerBlock);

    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float time_gpu = 0;
    cudaEventElapsedTime(&time_gpu, start_gpu, end_gpu);
    time_gpu /= 1000.0; // Convert to seconds

    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    if (isSorted(h_data, n)) {
        printf("GPU: Array is sorted correctly.\n");
    } else {
        printf("GPU: Array is NOT sorted correctly.\n");
    }

    // Print the GPU sorted array
    // printf("GPU sorted array:\n");
    // printArray(h_data, n);

    // Calculate and display speedup
    printf("Sequential Time: %.6f seconds\n", time_seq);
    printf("GPU Time: %.6f seconds\n", time_gpu);
    printf("Speedup: %.2fx\n", time_seq / time_gpu);

    // Cleanup
    cudaFree(d_data);
    free(h_data);
    free(h_data_seq);
    return 0;
}

