#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>
#include <assert.h>
#include <stdio.h>

double* CPUMatrix;
double* CPUMatrixTranspose;
double* hMatrix;
double* dMatrix;
double* hOutMatrix;
double* dOutMatrix;
unsigned int blocks;
unsigned int N = 1024;
unsigned int ELEMENTS = N * N;
unsigned int RANGE = 10;
unsigned int MAX_BLOCK_SIZE;
unsigned int BLOCK_SIZE = 32;
size_t matrixSize = sizeof(double) * ELEMENTS;

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %d, %s: %s", err, msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int getMaxThreadsPerBlock()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    checkCUDAError("cudaGetDeviceProperties");
    return prop.maxThreadsPerBlock;
}


double* initializeMatrixWithRandomNumbers(unsigned int range)
{
    double* matrix = new double[ELEMENTS];
    for (unsigned int i = 0; i < ELEMENTS; i++)
    {
        matrix[i] = rand() % range;
    }
    return matrix;
}

void printMatrix(char *header, double* matrix)
{
    std::cout << header << std::endl;
    for (unsigned int i = 0; i < N; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

void transposeMatrixHost(double* matrix, double* transpose)
{
    for (unsigned int i = 0; i < N; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            transpose[j * N + i] = matrix[i * N + j];
        }
    }
}

int checkCUDADevices()
{
    int devCnt;
    cudaGetDeviceCount(&devCnt);
    if (devCnt == 0)
    {
        perror("No CUDA devices available -- exiting.");
        return 1;
    }
}

void prologueGPU()
{
    cudaMalloc((void**)&dMatrix, matrixSize);
    cudaMalloc((void**)&dOutMatrix, matrixSize);
    checkCUDAError("cudaMalloc");
    cudaMemcpy(dMatrix, hMatrix, matrixSize, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpyHTD");
}

void epilogueGPU()
{
    cudaMemcpy(hOutMatrix, dOutMatrix, matrixSize, cudaMemcpyDeviceToHost);
}

void prologueCPU()
{
    hMatrix = initializeMatrixWithRandomNumbers(RANGE);
    hOutMatrix = initializeMatrixWithRandomNumbers(RANGE);
    CPUMatrixTranspose = initializeMatrixWithRandomNumbers(RANGE);
}

void epilogueCPU()
{
    delete hMatrix;
    delete hOutMatrix;
    delete CPUMatrixTranspose;
}

void checkEqualityOfOutputMatrices(double* host, double* device)
{
    for (unsigned int i = 0; i < N; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            assert(host[i * N + j] == device[i * N + j]);
        }
    }
}

__global__ void transposeMatrixDevice(const double* inMatrix, double* outMatrix, unsigned int BLOCK_SIZE, unsigned int N)
{
    int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int index_in = xIndex + N * yIndex;
    int index_out = yIndex + N * xIndex;

    outMatrix[index_out] = inMatrix[index_in];
    
}

int main()
{
    srand(time(NULL));
    checkCUDADevices();
    prologueCPU();
    prologueGPU();

    //printMatrix("Matrix before transpose:", hMatrix);

    transposeMatrixHost(hMatrix, CPUMatrixTranspose);

    //printMatrix("Matrix after transpose on host:", CPUMatrixTranspose);

    dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    transposeMatrixDevice << <grid, threads >> > (dMatrix, dOutMatrix, BLOCK_SIZE, N);
    cudaDeviceSynchronize();

    epilogueGPU();

    //transposeMatrixHost(CPUMatrix, CPUMatrixTranspose);
    //printMatrix("Matrix after transpose on device:", hOutMatrix);

    checkEqualityOfOutputMatrices(CPUMatrixTranspose, hOutMatrix);

    epilogueCPU();
    return 0;
}