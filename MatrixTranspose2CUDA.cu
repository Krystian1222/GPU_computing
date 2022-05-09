#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <chrono>
#include <fstream>

// N and Nd must be integral multiplies of BLOCK_SIZE
// BLOCK_SIZE <= 32 (maxThreadsPerBlock = 1024 on my device (sqrt(1024) = 32))

int getMaxThreadsPerGrid();
void checkCUDAError(const char* msg);

const unsigned int N = 192;
__device__ const unsigned int Nd = 192;
__device__ const unsigned int BLOCK_SIZE = 32;//getMaxThreadsPerGrid(); is not allowed
unsigned int ELEMENTS = N * N;
unsigned int RANGE = 100;
unsigned short int PRINT_MATRICES = 0;

struct Data
{
    long long int Z1kernelTime;
    long long int Z2kernelTime;
    long long int Z2kernelTimeWithoutSharedMemory;
    long long int CPUTime;
    int matrixSize;

    void printData()
    {
        std::cout << "\nMatrix size: " << this->matrixSize << "x" << this->matrixSize << std::endl;
        std::cout << "Kernel time with global memory (Z1):    " << this->Z1kernelTime << " nanoseconds." << std::endl;
        std::cout << "Kernel time with shared memory (Z2):    " << this->Z2kernelTime << " nanoseconds." << std::endl;
        std::cout << "Kernel time without shared memory (Z2): " << this->Z2kernelTimeWithoutSharedMemory << " nanoseconds." << std::endl;
        std::cout << "CPU Time:                               " << this->CPUTime << " nanoseconds." << std::endl;
    }

    void saveDataToFile()
    {
        std::ofstream statFile("L10_Z2.csv", std::ios::app);
        if (statFile.good() == true)
        {
            statFile << this->matrixSize << "x" << this->matrixSize << ";"
                << this->Z1kernelTime << ";"
                << this->Z2kernelTime << ";"
                << this->Z2kernelTimeWithoutSharedMemory << ";"
                << this->CPUTime << std::endl;

            std::cout << "Writing data to file: completed." << std::endl;
        }
        else
        {
            std::cout << "Writing data to file: error." << std::endl;
        }
        statFile.close();
    }
};

Data* data = (Data*)malloc(sizeof(Data));

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %d, %s: %s", err, msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int getMaxThreadsPerGrid()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    checkCUDAError("cudaGetDeviceProperties");
    return floor(sqrt(prop.maxThreadsPerBlock));
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

long long int getChronoTime(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
{
    std::chrono::steady_clock::duration time_span = end - start;
    long long int nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_span).count();
    return nanoseconds;
}

namespace Z1
{
    float* CPUMatrix;
    float* CPUMatrixTranspose;
    float* hMatrix;
    float* dMatrix;
    float* hOutMatrix;
    //float* dOutMatrix;
    size_t matrixSize = sizeof(float) * ELEMENTS;

    float* initializeMatrixWithRandomNumbers(unsigned int range)
    {
        float* matrix = new float[ELEMENTS];
        for (unsigned int i = 0; i < ELEMENTS; i++)
        {
            matrix[i] = rand() % range;
        }
        return matrix;
    }

    void printMatrix(char* header, float* matrix)
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

    void transposeMatrixHost(float* matrix, float* transpose)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            for (unsigned int j = 0; j < N; j++)
            {
                transpose[j * N + i] = matrix[i * N + j];
            }
        }
    }

    void prologueGPU()
    {
        cudaMalloc((void**)&dMatrix, matrixSize);
        //cudaMalloc((void**)&dOutMatrix, matrixSize);
        checkCUDAError("cudaMalloc");
        cudaMemcpy(dMatrix, hMatrix, matrixSize, cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpyHTD");
    }

    void epilogueGPU()
    {
        cudaMemcpy(hOutMatrix, dMatrix, matrixSize, cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpyDTH");
        cudaFree(dMatrix);
        //cudaFree(dOutMatrix);
        checkCUDAError("cudaFree");
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

    void checkEqualityOfOutputMatrices(float* host, float* device)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            for (unsigned int j = 0; j < N; j++)
            {
                //printf("%d\n", device[i * N + j]);
                assert(host[i * N + j] == device[i * N + j]);
            }
        }
    }

    __global__ void transposeMatrixDevice(float* inMatrix)
    {
        int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        inMatrix[Nd * x + y] = inMatrix[Nd * y + x];
#if __CUDART__ >= 200
        __syncthreads();
#endif
    }
}

namespace Z2
{
    typedef struct
    {
        int N;
        int stride;
        float* elements;
    } Matrix;

    float* initializeMatrixWithRandomNumber(Matrix A, unsigned int range)
    {
        A.elements = new float[ELEMENTS];
        for (unsigned int i = 0; i < ELEMENTS; i++)
        {
            A.elements[i] = rand() % range;
        }
        return A.elements;
    }

    float* initializeMatrixWithZeros(Matrix A)
    {
        A.elements = new float[ELEMENTS];
        for (unsigned int i = 0; i < ELEMENTS; i++)
        {
            A.elements[i] = 0;
        }
        return A.elements;
    }

    void printMatrix(char* header, Matrix A)
    {
        std::cout << header << std::endl;
        for (unsigned int i = 0; i < N; i++)
        {
            for (unsigned int j = 0; j < N; j++)
            {
                std::cout << A.elements[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void transposeMatrixHost(Matrix matrix, Matrix transpose)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            for (unsigned int j = 0; j < N; j++)
            {
                transpose.elements[j * N + i] = matrix.elements[i * N + j];
            }
        }
    }

    void checkEqualityOfOutputMatrices(Matrix host, Matrix device)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            for (unsigned int j = 0; j < N; j++)
            {
                assert(host.elements[i * N + j] == device.elements[i * N + j]);
            }
        }
    }

    __device__ float GetElement(const Matrix A, int row, int col)
    {
        return A.elements[row * A.stride + col];
    }

    __device__ void SetElement(Matrix A, int row, int col, int value)
    {
        A.elements[row * A.stride + col] = value;
    }

    __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
    {
        Matrix Asub;
        Asub.N = BLOCK_SIZE;
        Asub.stride = A.stride;
        Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];

        return Asub;
    }

    __global__ void MatTransposeKernel(const Matrix A, const Matrix B)
    {
        int blockRow = blockIdx.y;

        int row = threadIdx.y;
        int col = threadIdx.x;

        for (int m = 0; m < (A.N / BLOCK_SIZE); ++m)
        {
            Matrix Asub = GetSubMatrix(A, blockRow, m);
            Matrix Bsub = GetSubMatrix(B, m, blockRow);
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
            As[col][row] = GetElement(Asub, col, row);
            SetElement(Bsub, row, col, As[col][row]);
#if __CUDART__ >= 200
            __syncthreads();
#endif
        }
    }

    __global__ void MatTransposeKernelWithoutSharedMemory(const Matrix A, const Matrix B)
    {
        int blockRow = blockIdx.y;

        int row = threadIdx.y;
        int col = threadIdx.x;

        for (int m = 0; m < (A.N / BLOCK_SIZE); ++m)
        {
            Matrix Asub = GetSubMatrix(A, blockRow, m);
            Matrix Bsub = GetSubMatrix(B, m, blockRow);
            float As[BLOCK_SIZE][BLOCK_SIZE];
            As[col][row] = GetElement(Asub, col, row);
            SetElement(Bsub, row, col, As[col][row]);
#if __CUDART__ >= 200
            __syncthreads();
#endif
        }
    }

    void MatTranspose(const Matrix A, Matrix B, Matrix C)
    {
        Matrix d_A;
        d_A.N = d_A.stride = A.N;
        size_t size = A.N * A.N * sizeof(int);
        cudaMalloc((void**)&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

        Matrix d_B;
        d_B.N = d_B.stride = B.N;
        size = B.N * B.N * sizeof(int);
        cudaMalloc((void**)&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

        Matrix d_C;
        d_C.N = d_C.stride = C.N;
        size = C.N * C.N * sizeof(int);
        cudaMalloc((void**)&d_C.elements, size);
        cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(B.N / dimBlock.x, A.N / dimBlock.y);

        auto sK = std::chrono::steady_clock::now();
        MatTransposeKernel << <dimGrid, dimBlock >> > (d_A, d_B);
        cudaDeviceSynchronize();
        auto eK = std::chrono::steady_clock::now();
        data->Z2kernelTime = getChronoTime(sK, eK);

        auto sKWOSM = std::chrono::steady_clock::now();
        MatTransposeKernelWithoutSharedMemory << <dimGrid, dimBlock >> > (d_A, d_C);
        cudaDeviceSynchronize();
        auto eKWOSM = std::chrono::steady_clock::now();
        data->Z2kernelTimeWithoutSharedMemory = getChronoTime(sKWOSM, eKWOSM);

        cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
    }
}

int main()
{
    srand(time(NULL));
    checkCUDADevices();

    {
        using namespace Z1;
        data->matrixSize = N;
        prologueCPU();
        prologueGPU();
        transposeMatrixHost(hMatrix, CPUMatrixTranspose);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        unsigned int gridSize = N / dimBlock.x;
        dim3 dimGrid(gridSize, gridSize);

        auto sK = std::chrono::steady_clock::now();
        transposeMatrixDevice << <dimGrid, dimBlock >> > (dMatrix);
        cudaDeviceSynchronize();
        auto eK = std::chrono::steady_clock::now();

        data->Z1kernelTime = getChronoTime(sK, eK);

        epilogueGPU();
        if (PRINT_MATRICES == 1)
        {
            printMatrix("Matrix before transpose:", hMatrix);
            printMatrix("Matrix after transpose on host:", CPUMatrixTranspose);
            printMatrix("Matrix after transpose on device:", hOutMatrix);
        }
        checkEqualityOfOutputMatrices(CPUMatrixTranspose, hOutMatrix);
        epilogueCPU();
    }

    {
        using namespace Z2;
        Matrix A, B, C, BeforeTCPU, AfterTCPU;
        A.N = N;
        A.stride = N;
        A.elements = initializeMatrixWithRandomNumber(A, RANGE);

        B.N = N;
        B.stride = N;
        B.elements = initializeMatrixWithZeros(B);

        C.N = N;
        C.stride = N;
        C.elements = initializeMatrixWithZeros(C);

        BeforeTCPU.N = N;
        BeforeTCPU.stride = N;
        BeforeTCPU.elements = A.elements;

        AfterTCPU.N = N;
        AfterTCPU.stride = N;
        AfterTCPU.elements = initializeMatrixWithZeros(B);

        auto sH = std::chrono::steady_clock::now();
        transposeMatrixHost(BeforeTCPU, AfterTCPU);
        auto eH = std::chrono::steady_clock::now();

        data->CPUTime = getChronoTime(sH, eH);

        MatTranspose(A, B, C);

        if (PRINT_MATRICES == 1)
        {
            printMatrix("Matrix A", A);
            printMatrix("Matrix B", B);
            printMatrix("Before TCPU", BeforeTCPU);
            printMatrix("After TCPU", AfterTCPU);
        }
        checkEqualityOfOutputMatrices(AfterTCPU, B);
        checkEqualityOfOutputMatrices(AfterTCPU, C);
        data->printData();
        //data->saveDataToFile();
    }

    return 0;
}