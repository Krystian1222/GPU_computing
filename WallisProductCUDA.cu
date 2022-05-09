#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>

#define PRECISION 15

double* hArray;
double* dArray;
double* CPUArray;
double* piNumberArrayHost;
double* piNumberArrayDevice; 
double* piNumberForDevice;
double* piNumberForHost;
double* A;
double* B;
double* tableOfTermsOnHost;
double* valuesOnDevice;
double* valuesOnHost;

int blocks;
std::ofstream statFile("L08_Z2.csv", std::ios::app);

struct Data
{
    int printFlag = 0;
    int saveDataFlag = 0;
    int N = 0;
    double eps = 0;
    long long int GPUTime = 0;
    long long int CPUTime = 0;
    long long int HostToDeviceTime = 0;
    long long int DeviceToHostTime = 0;
    long long int differenceBetweenHostTimeAndDeviceTime = 0;
    long long int termsMultiplicationOnHostTime = 0;
    long long int termsMultiplicationOnDeviceTime = 0;
    long long int computeTermsOnDevice = 0;
    long long int termsOnDeviceMultiplyingOnHostTime = 0;

    void saveDataToFile()
    {
        if (statFile.good() == true)
        {
            statFile << this->N << ";"
                     << this->eps << ";"
                     << this->HostToDeviceTime << ";"
                     << this->DeviceToHostTime << ";"
                     << this->computeTermsOnDevice << ";"
                     << this->termsMultiplicationOnDeviceTime << ";"
                     << this->GPUTime << ";"
                     << this->CPUTime << ";"
                     << this->termsMultiplicationOnHostTime << ";"
                     << this->termsOnDeviceMultiplyingOnHostTime << ";"
                     << this->differenceBetweenHostTimeAndDeviceTime << ";"
                     << std::endl;

            std::cout << "Writing data to file: completed." << std::endl;
        }
        else
        {
            std::cout << "Writing data to file: error." << std::endl;
        }
    }

    void printData()
    {
        std::cout << "\nN: " << this->N << std::endl;
        std::cout << "eps: " << this->eps << std::endl;
        std::cout << "Times:" << std::endl;
        std::cout << "\tTransfer host to device: " << this->HostToDeviceTime << " nanoseconds." << std::endl;
        std::cout << "\tTransfer device to host: " << this->DeviceToHostTime << " nanoseconds." << std::endl;
        std::cout << "\tComputing terms on device: " << this->computeTermsOnDevice << " nanoseconds." << std::endl;
        std::cout << "\tGPU summary (data transfer + computing terms + multiplying terms): " << this->GPUTime << " nanoseconds." << std::endl;
        std::cout << "\t\tTerms multiplication on device: " << this->termsMultiplicationOnDeviceTime << " nanoseconds." << std::endl;
        std::cout << "\tCPU: " << this->CPUTime << " nanoseconds." << std::endl;
        std::cout << "\tGPU + CPU (data transfer + computing terms on device + multiplying terms on host): " << this->termsOnDeviceMultiplyingOnHostTime << " nanoseconds." << std::endl;
        std::cout << "\t\tTerms multiplication on host: " << this->termsMultiplicationOnHostTime << " nanoseconds." << std::endl;
        std::cout << "\tCPU - GPU: " << this->differenceBetweenHostTimeAndDeviceTime << " nanoseconds." << std::endl;
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

long long int getChronoTime(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
{
    std::chrono::steady_clock::duration time_span = end - start;
    long long int nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_span).count();
    return nanoseconds;
}

void prologue(int N) {
    cudaMalloc((void**)&valuesOnDevice, sizeof(double) * data->N);
    cudaMalloc((void**)&dArray, sizeof(double) * N);
    cudaMalloc((void**)&piNumberArrayDevice, sizeof(double) * N);
    cudaMalloc((void**)&piNumberForDevice, sizeof(double));
    checkCUDAError("cudaMalloc");

    auto startHostToDeviceTime = std::chrono::steady_clock::now();
    cudaMemcpy(dArray, hArray, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(piNumberArrayDevice, piNumberArrayHost, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(piNumberForDevice, piNumberForHost, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(valuesOnDevice, valuesOnHost, sizeof(double) * data->N, cudaMemcpyHostToDevice);
    auto endHostToDeviceTime = std::chrono::steady_clock::now();
    data->HostToDeviceTime = getChronoTime(startHostToDeviceTime, endHostToDeviceTime);

    checkCUDAError("cudaMemcpyHTD");
}

void epilogue(int N) {
    auto startDeviceToHostTime = std::chrono::steady_clock::now();
    cudaMemcpy(piNumberArrayHost, piNumberArrayDevice, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(piNumberForHost, piNumberForDevice, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(valuesOnHost, valuesOnDevice, sizeof(double) * N, cudaMemcpyDeviceToHost);
    auto endDeviceToHostTime = std::chrono::steady_clock::now();
    data->DeviceToHostTime = getChronoTime(startDeviceToHostTime, endDeviceToHostTime);
    checkCUDAError("cudaMemcpyDTH");

    cudaFree(dArray);
    cudaFree(piNumberArrayDevice);
    cudaFree(piNumberForDevice);
    cudaFree(valuesOnDevice);
    checkCUDAError("cudaFree");
}

void prologueCPU(long long int N)
{
    valuesOnHost = (double*)malloc(sizeof(double) * data->N);
    for (int i = 0; i < data->N; i++)
    {
        valuesOnHost[i] = i + 1;
    }
    CPUArray = (double*)malloc(sizeof(double) * N);
    hArray = (double*)malloc(sizeof(double) * N);
    piNumberArrayHost = (double*)malloc(sizeof(double) * N);
    piNumberForHost = (double*)malloc(sizeof(double));
    A = (double*)malloc(sizeof(double) * N);
    B = (double*)malloc(sizeof(double) * N);
    tableOfTermsOnHost = (double*)malloc(sizeof(double) * N);

    memset(hArray, 0, sizeof(double) * N);
    memset(piNumberArrayHost, 0, sizeof(double) * N);
    memset(piNumberForHost, 0, sizeof(double));
    *piNumberForHost = 1.0;
    memset(CPUArray, 0, sizeof(double) * N);
    memset(A, 0, sizeof(double) * N);
    memset(B, 0, sizeof(double) * N);
    memset(tableOfTermsOnHost, 0, sizeof(double) * N);
}

void epilogueCPU(void)
{
    free(CPUArray);
    free(hArray);
    free(piNumberForHost);
    free(piNumberArrayHost);
    free(A);
    free(B);
    free(tableOfTermsOnHost);
    free(valuesOnHost);
}

int getMaxThreadsPerBlock()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    checkCUDAError("cudaGetDeviceProperties");
    return prop.maxThreadsPerBlock;
}

__device__ double atomicMul(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do 
    {
        assumed = old;
#if __CUDA_ARCH__ >= 200
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed)));
#endif
    } while (assumed != old);
#if __CUDA_ARCH__ >= 200
    return __longlong_as_double(old);
#endif
}

__global__ void computeTermsDevice(double* A, int N)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < N && x > 0)
    {
        A[x - 1] = (4.0 * pow(x, 2.0)) / (4.0 * pow(x, 2.0) - 1.0);
    }
}

__global__ void multiplyingTermsDevice(double* d_a, double* d_out, int N, double eps, double* valuesOnDevice)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int flag = 1;
    if (x < N - 1)
    {
        if (x > 1 && (abs(valuesOnDevice[x] - valuesOnDevice[x - 1]) < eps))
        {
            flag = 0;
        }
        if (flag == 1)
        {
            atomicMul(d_out, d_a[x]);
            valuesOnDevice[x] = *d_out * 2.0;
        }
    }
}

double multiplyingTermsFromDeviceOnHost(double* A, double *B, double eps, int N)
{
    double product = 1.0;
    double piNumber = 0.0;
    for (int i = 0; i < N - 1; i++)
    {
        product *= A[i];
        B[i] = product * 2.0;
        if (i > 1 && (abs(B[i] - B[i - 1]) < eps))
        {
            piNumber = B[i];
            return piNumber;
        }
    }

    std::cout << "N = " << N << " is too low for given eps = " << std::fixed << std::setprecision(PRECISION) << eps << std::endl;
    piNumber = B[N - 2];
    return piNumber;
}

double computeTermsAndMultiplyingTermsAllOnHost(double *A, double* B, double eps, int N)
{
    for (int i = 1; i < N; i++)
    {
        A[i - 1] = (4.0 * pow(i, 2.0)) / (4.0 * pow(i, 2.0) - 1.0);
    }

    double product = 1.0;
    double piNumber = 0.0;
 
    for (int i = 0; i < N - 1; i++)
    {
        product *= A[i];
        B[i] = product * 2.0;
        if (i > 1 && (abs(B[i] - B[i - 1]) < eps))
        {
            piNumber = B[i];
            return piNumber;
        }
    }

    std::cout << "N = " << N << " is too low for given eps = " << std::fixed << std::setprecision(PRECISION) << eps << std::endl;
    piNumber = B[N - 2];

    return piNumber;
}

void printPiNumbers(double piDevice, double piHost, double piFromHostTermsFromDevice)
{
    std::cout << "PI number - computing terms on device and multiplying terms on device:" << std::endl;
    std::cout << std::fixed << std::setprecision(PRECISION) << piDevice << std::endl;

    std::cout << "PI number - computing terms on host and multiplying terms on host:" << std::endl;
    std::cout << std::fixed << std::setprecision(PRECISION) << piHost << std::endl;

    std::cout << "PI number - computing terms on device and multiplying terms on host:" << std::endl;
    std::cout << std::fixed << std::setprecision(PRECISION) << piFromHostTermsFromDevice << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "\nWrong number of arguments." << std::endl;
        return 1;
    }
    char* eptr;
    data->N = atoi(argv[1]);
    data->eps = strtod(argv[2], &eptr);
    data->printFlag = atoi(argv[3]);
    data->saveDataFlag = atoi(argv[4]);
    
    int devCnt;
    cudaGetDeviceCount(&devCnt);
    if (devCnt == 0)
    {
        perror("No CUDA devices available -- exiting.");
        return 1;
    }

    int MAX_BLOCK_SIZE = getMaxThreadsPerBlock();

    blocks = data->N / MAX_BLOCK_SIZE;
    if (data->N % MAX_BLOCK_SIZE)
        blocks++;

    prologueCPU(data->N);

    prologue(data->N);

/***********************************************************************************************/
// COMPUTING TERMS OF WALLIS PRODUCT USING DEVICE
    auto startKernelTerms = std::chrono::steady_clock::now();
    computeTermsDevice << <blocks, MAX_BLOCK_SIZE >> > (dArray, data->N);
    cudaDeviceSynchronize();
    auto endKernelTerms = std::chrono::steady_clock::now();
    data->computeTermsOnDevice = getChronoTime(startKernelTerms, endKernelTerms);   
/**********************************************************************************************/

/**********************************************************************************************/
// MULTIPLYING TERMS ON HOST
    auto termsOnHostStart = std::chrono::steady_clock::now();
    cudaMemcpy(tableOfTermsOnHost, dArray, sizeof(double) * data->N, cudaMemcpyDeviceToHost);
    auto termsOnHostEnd = std::chrono::steady_clock::now();
    data->DeviceToHostTime += getChronoTime(termsOnHostStart, termsOnHostEnd);

    auto startComputePiNumberFromTermsFromDevice = std::chrono::steady_clock::now();
    double piNumberFromTermsFromDevice = multiplyingTermsFromDeviceOnHost(tableOfTermsOnHost, B, data->eps, data->N);
    auto endComputePiNumberFromTermsFromDevice = std::chrono::steady_clock::now();
    data->termsMultiplicationOnHostTime = getChronoTime(startComputePiNumberFromTermsFromDevice, endComputePiNumberFromTermsFromDevice);
    data->termsOnDeviceMultiplyingOnHostTime = data->computeTermsOnDevice + data->termsMultiplicationOnHostTime;
/**********************************************************************************************/

/**********************************************************************************************/
// MULTIPLYING TERMS ON DEVICE
    auto startComputePiNumberDevice = std::chrono::steady_clock::now();
    multiplyingTermsDevice << <blocks, MAX_BLOCK_SIZE >> > (dArray, piNumberForDevice, data->N, data->eps, valuesOnDevice);
    cudaDeviceSynchronize();
    checkCUDAError("multiplyingTermsDevice");
    auto endComputePiNumberDevice = std::chrono::steady_clock::now();
    
    auto startPiNumberDTH = std::chrono::steady_clock::now();
    cudaMemcpy(piNumberForHost, piNumberForDevice, sizeof(double), cudaMemcpyDeviceToHost);
    auto endPiNumberDTH = std::chrono::steady_clock::now();
    data->DeviceToHostTime += getChronoTime(startPiNumberDTH, endPiNumberDTH);
    
    data->termsMultiplicationOnDeviceTime = getChronoTime(startComputePiNumberDevice, endComputePiNumberDevice);
    double piNumberGeneratedByDevice = 2.0 * (*piNumberForHost);
/**********************************************************************************************/

/**********************************************************************************************/
// COMPUTE PI NUMBER ON HOST (COMPUTE TERMS AND MULTIPLYING)
    auto startHost = std::chrono::steady_clock::now();
    double piNumber = computeTermsAndMultiplyingTermsAllOnHost(A, B, data->eps, data->N);
    auto endHost = std::chrono::steady_clock::now();
/**********************************************************************************************/

    epilogue(data->N);

    data->CPUTime = getChronoTime(startHost, endHost);
    data->GPUTime = data->DeviceToHostTime + data->HostToDeviceTime + data->computeTermsOnDevice + data->termsMultiplicationOnDeviceTime;
    data->differenceBetweenHostTimeAndDeviceTime = data->CPUTime - data->GPUTime;
    data->termsOnDeviceMultiplyingOnHostTime = data->DeviceToHostTime + data->HostToDeviceTime + data->computeTermsOnDevice + data->termsMultiplicationOnHostTime;

    if (data->printFlag == 1)
    {
        data->printData();
    }
    if (data->saveDataFlag == 1)
    {
        data->saveDataToFile();
    }

    printPiNumbers(piNumberGeneratedByDevice, piNumber, piNumberFromTermsFromDevice);

    epilogueCPU();
    statFile.close();
    free(data);
    return 0;
}
