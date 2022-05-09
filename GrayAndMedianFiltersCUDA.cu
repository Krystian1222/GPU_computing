#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "qdbmp.c"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <assert.h>

BMP* hBMPGray;
BMP* hBMPMedian;
BMP* CPUBMPGray;
BMP* CPUBMPMedian;
BMP* hBMPMedianOut;
BMP* CPUBMPMedianOut;

UCHAR* dBMPGrayData;
UCHAR* hBMPGrayData;
UCHAR* dBMPMedianData;
UCHAR* hBMPMedianData;
UCHAR* dBMPMedianDataOut;
UCHAR* hBMPMedianDataOut;
UCHAR* CPUBMPGrayData;
UCHAR* CPUBMPMedianData;
UCHAR* CPUBMPMedianDataOut;

int blocks;
int MAX_BLOCK_SIZE;
std::ofstream statFile("L09_Z2.csv", std::ios::app);

struct Data
{
	int printFlag = 0;
	int saveDataFlag = 0;
	int imgWidth = 0;
	int imgHeight = 0;
	size_t size = 0;
	char* filename = NULL;
	long long int GPUGrayTime = 0;
	long long int GPUMedianTime = 0;
	long long int kernelGrayTime = 0;
	long long int kernelMedianTime = 0;
	long long int CPUGrayTime = 0;
	long long int CPUMedianTime = 0;
	long long int diffBetweenHostAndDeviceGray = 0;
	long long int diffBetweenHostAndDeviceMedian = 0;
	long long int HTDGrayTime = 0;
	long long int HTDMedianTime = 0;
	long long int DTHGrayTime = 0;
	long long int DTHMedianTime = 0;

	void setAllocSize()
	{
		this->size = this->imgWidth * this->imgHeight * 3;
	}

	void saveDataToFile()
	{
		if (this->saveDataFlag == 1)
		{
			if (statFile.good() == true)
			{
				statFile << this->filename << ";"
					<< this->imgWidth << "x" << this->imgHeight << ";"
					<< this->HTDGrayTime << ";"
					<< this->DTHGrayTime << ";"
					<< this->kernelGrayTime << ";"
					<< this->GPUGrayTime << ";"
					<< this->CPUGrayTime << ";"
					<< this->diffBetweenHostAndDeviceGray << ";"
					<< this->HTDMedianTime << ";"
					<< this->DTHMedianTime << ";"
					<< this->kernelMedianTime << ";"
					<< this->GPUMedianTime << ";"
					<< this->CPUMedianTime << ";"
					<< this->diffBetweenHostAndDeviceMedian << ";"
					<< std::endl;

				std::cout << "\nWriting data to file: completed." << std::endl;
			}
			else
			{
				std::cout << "\nWriting data to file: error." << std::endl;
			}
		}
	}

	void printData()
	{
		if (this->printFlag == 1)
		{
			std::cout << "\nFilename: " << this->filename << std::endl;
			std::cout << "Image size: " << this->imgWidth << "x" << this->imgHeight << " [px]." << std::endl;
			std::cout << "Convert to gray scale:" << std::endl;
			std::cout << "\tTransfer host to device: " << this->HTDGrayTime << " nanoseconds." << std::endl;
			std::cout << "\tTransfer device to host: " << this->DTHGrayTime << " nanoseconds." << std::endl;
			std::cout << "\tKernel: " << this->kernelGrayTime << " nanoseconds." << std::endl;
			std::cout << "\tGPU summary (data transfer + kernel): " << this->GPUGrayTime << " nanoseconds." << std::endl;
			std::cout << "\tCPU: " << this->CPUGrayTime << " nanoseconds." << std::endl;
			std::cout << "\tCPU - GPU: " << this->diffBetweenHostAndDeviceGray << " nanoseconds." << std::endl;
			std::cout << "\nMedian filter:" << std::endl;
			std::cout << "\tTransfer host to device: " << this->HTDMedianTime << " nanoseconds." << std::endl;
			std::cout << "\tTransfer device to host: " << this->DTHMedianTime << " nanoseconds." << std::endl;
			std::cout << "\tKernel: " << this->kernelMedianTime << " nanoseconds." << std::endl;
			std::cout << "\tGPU summary (data transfer + kernel): " << this->GPUMedianTime << " nanoseconds." << std::endl;
			std::cout << "\tCPU: " << this->CPUMedianTime << " nanoseconds." << std::endl;
			std::cout << "\tCPU - GPU: " << this->diffBetweenHostAndDeviceMedian << " nanoseconds." << std::endl;
		}
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

void checkEqualityOfOutputImages(BMP* gpuImg, BMP* cpuImg)
{
	UCHAR gR, gG, gB, cR, cG, cB;
	for (int x = 0; x < data->imgWidth; x++)
	{
		for (int y = 0; y < data->imgHeight; y++)
		{
			BMP_GetPixelRGB(gpuImg, x, y, &gR, &gG, &gB);
			BMP_GetPixelRGB(cpuImg, x, y, &cR, &cG, &cB);
			assert(gR == cR);
			assert(gG == cG);
			assert(gB == cB);
		}
	}
}

int prologueCPU(char* filename)
{
	data->filename = filename;
	hBMPGray = BMP_ReadFile(filename);
	BMP_CHECK_ERROR(stdout, -1);
	hBMPMedian = BMP_ReadFile(filename);
	BMP_CHECK_ERROR(stdout, -1);
	CPUBMPGray = BMP_ReadFile(filename);
	BMP_CHECK_ERROR(stdout, -1);
	CPUBMPMedian = BMP_ReadFile(filename);
	BMP_CHECK_ERROR(stdout, -1);
	hBMPMedianOut = BMP_ReadFile(filename);
	BMP_CHECK_ERROR(stdout, -1);
	CPUBMPMedianOut = BMP_ReadFile(filename);
	BMP_CHECK_ERROR(stdout, -1);

	data->imgWidth = BMP_GetWidth(hBMPGray);
	data->imgHeight = BMP_GetHeight(hBMPGray);

	data->setAllocSize();

	hBMPGrayData = hBMPGray->Data;
	hBMPMedianData = hBMPMedian->Data;
	CPUBMPGrayData = CPUBMPGray->Data;
	CPUBMPMedianData = CPUBMPMedian->Data;
	hBMPMedianDataOut = hBMPMedianOut->Data;
	CPUBMPMedianDataOut = CPUBMPMedianOut->Data;
}

void prologueGPU()
{
	cudaMalloc((void**)&dBMPGrayData, data->size);
	cudaMalloc((void**)&dBMPMedianData, data->size);
	cudaMalloc((void**)&dBMPMedianDataOut, data->size);
	checkCUDAError("cudaMalloc");

	auto startHTDGrayTime = std::chrono::steady_clock::now();
	cudaMemcpy(dBMPGrayData, hBMPGrayData, data->size, cudaMemcpyHostToDevice);
	auto endHTDGrayTime = std::chrono::steady_clock::now();
	data->HTDGrayTime = getChronoTime(startHTDGrayTime, endHTDGrayTime);

	auto startHTDMedianTime = std::chrono::steady_clock::now();
	cudaMemcpy(dBMPMedianData, hBMPMedianData, data->size, cudaMemcpyHostToDevice);
	cudaMemcpy(dBMPMedianDataOut, hBMPMedianDataOut, data->size, cudaMemcpyHostToDevice);
	auto endHTDMedianTime = std::chrono::steady_clock::now();
	data->HTDMedianTime = getChronoTime(startHTDMedianTime, endHTDMedianTime);
	checkCUDAError("cudaMemcpyHTD");
}

int epilogueCPU(char* grayGPU, char* grayCPU, char* medianGPU, char *medianCPU)
{
	hBMPGray->Data = hBMPGrayData;
	hBMPMedian->Data = hBMPMedianData;
	CPUBMPGray->Data = CPUBMPGrayData;
	CPUBMPMedian->Data = CPUBMPMedianData;
	hBMPMedianOut->Data = hBMPMedianDataOut;
	CPUBMPMedianOut->Data = CPUBMPMedianDataOut;

	checkEqualityOfOutputImages(hBMPGray, CPUBMPGray);
	checkEqualityOfOutputImages(hBMPMedianOut, CPUBMPMedianOut);

	BMP_WriteFile(hBMPGray, grayGPU);
	BMP_CHECK_ERROR(stdout, -2);
	BMP_WriteFile(CPUBMPGray, grayCPU);
	BMP_CHECK_ERROR(stdout, -2);
	BMP_WriteFile(hBMPMedianOut, medianGPU);
	BMP_CHECK_ERROR(stdout, -2);
	BMP_WriteFile(CPUBMPMedianOut, medianCPU);
	BMP_CHECK_ERROR(stdout, -2);

	BMP_Free(hBMPGray);
	BMP_Free(hBMPMedian);
	BMP_Free(CPUBMPGray);
	BMP_Free(CPUBMPMedian);
	BMP_Free(hBMPMedianOut);
	BMP_Free(CPUBMPMedianOut);
}

void epilogueGPU()
{
	auto startDTHGray = std::chrono::steady_clock::now();
	cudaMemcpy(hBMPGrayData, dBMPGrayData, data->size, cudaMemcpyDeviceToHost);
	auto endDTHGray = std::chrono::steady_clock::now();
	data->DTHGrayTime = getChronoTime(startDTHGray, endDTHGray);

	auto startDTHMedian = std::chrono::steady_clock::now();
	cudaMemcpy(hBMPMedianData, dBMPMedianData, data->size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hBMPMedianDataOut, dBMPMedianDataOut, data->size, cudaMemcpyDeviceToHost);
	auto endDTHMedian = std::chrono::steady_clock::now();
	data->DTHMedianTime = getChronoTime(startDTHMedian, endDTHMedian);
	checkCUDAError("cudaMemcpyDTH");

	cudaFree(dBMPGrayData);
	cudaFree(dBMPMedianData);
	cudaFree(dBMPMedianDataOut);
	checkCUDAError("cudaFree");
}

int getMaxThreadsPerBlock()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	checkCUDAError("cudaGetDeviceProperties");
	return prop.maxThreadsPerBlock;
}

void BMP_GetPixelRGBHost(UCHAR* data, int height, int width, UINT x, UINT y, UCHAR* r, UCHAR* g, UCHAR* b)
{
	UCHAR bytes_per_pixel = 3;
	UCHAR* pixel = data + ((height - y - 1) * bytes_per_pixel * width + x * bytes_per_pixel);

	if (r)	*r = *(pixel + 2);
	if (g)	*g = *(pixel + 1);
	if (b)	*b = *(pixel + 0);
}

void BMP_SetPixelRGBHost(UCHAR *data, int height, int width, UINT x, UINT y, UCHAR r, UCHAR g, UCHAR b)
{
	UCHAR bytes_per_pixel = 3;
	UCHAR* pixel = data + ((height - y - 1) * bytes_per_pixel * width + x * bytes_per_pixel);

	*(pixel + 2) = r;
	*(pixel + 1) = g;
	*(pixel + 0) = b;
}

__device__ void BMP_GetPixelRGBDevice(UCHAR* data, int height, int width, UINT x, UINT y, UCHAR* r, UCHAR* g, UCHAR* b)
{
	UCHAR bytes_per_pixel = 3;
	UCHAR* pixel = data + ((height - y - 1) * bytes_per_pixel * width + x * bytes_per_pixel);

	if (r)	*r = *(pixel + 2);
	if (g)	*g = *(pixel + 1);
	if (b)	*b = *(pixel + 0);
}

__device__ void BMP_SetPixelRGBDevice(UCHAR* data, int height, int width, UINT x, UINT y, UCHAR r, UCHAR g, UCHAR b)
{
	UCHAR bytes_per_pixel = 3;
	UCHAR* pixel = data + ((height - y - 1) * bytes_per_pixel * width + x * bytes_per_pixel);

	*(pixel + 2) = r;
	*(pixel + 1) = g;
	*(pixel + 0) = b;
}

void cvt2GSHost(UCHAR *data, int width, int height)
{
	UCHAR r, g, b;
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			BMP_GetPixelRGBHost(data, height, width, x, y, &r, &g, &b);
			int gray = (r + g + b) / 3;
			BMP_SetPixelRGBHost(data, height, width, x, y, gray, gray, gray);
		}
	}
}

__global__ void cvt2GSDevice(UCHAR* data, int width, int height)
{
	UCHAR r, g, b;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x = idx % width;
	int y = (idx - x) / width;

	if (x < width && y < height && x >= 0 && y >= 0)
	{
		BMP_GetPixelRGBDevice(data, height, width, x, y, &r, &g, &b);
		int gray = (r + g + b) / 3;
		BMP_SetPixelRGBDevice(data, height, width, x, y, gray, gray, gray);
	}
}

unsigned char medianHost(unsigned char* c, int size)
{
	unsigned char a = 0;
	for (unsigned char i = 0; i < size; i++)
	{
		for (unsigned char j = i + 1; j < size; j++)
		{
			if (c[i] > c[j])
			{
				a = c[i];
				c[i] = c[j];
				c[j] = a;
			}
		}
	}
	return c[4];
}

__device__ unsigned char medianDevice(unsigned char* c, int size)
{
	unsigned char a = 0;
	for (unsigned char i = 0; i < size; i++)
	{
		for (unsigned char j = i + 1; j < size; j++)
		{
			if (c[i] > c[j])
			{
				a = c[i];
				c[i] = c[j];
				c[j] = a;
			}
		}
	}

	return c[4];
}

void medianFilterHost(UCHAR* imgIn, UCHAR* imgOut, int width, int height)
{
	unsigned char rval[9], gval[9], bval[9], m;
	int size = 3;
	int margin = (size - 1) / 2;

	for (int i = margin; i < (width - margin); i++)
	{
		for (int j = margin; j < (height - margin); j++)
		{
			m = 0;
			for (int k = 0; k < size; k++)
			{
				for (int l = 0; l < size; l++)
				{
					BMP_GetPixelRGBHost(imgIn, height, width, i + k - margin, j + l - margin, &rval[m], &gval[m], &bval[m]);
					m++;
				}
			}
			BMP_SetPixelRGBHost(imgOut, height, width, i, j, medianHost(rval, 9), medianHost(gval, 9), medianHost(bval, 9));
		}
	}
}

__global__ void medianFilterDevice(UCHAR *imgIn, UCHAR *imgOut, int width, int height)
{
	unsigned char rval[9], gval[9], bval[9], m;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x = idx % width;
	int y = (idx - x) / width;
	int margin = 1;

	if (x < width - margin && y < height - margin && x >= margin && y >= margin)
	{
		m = 0;
		for (int k = 0; k < 3; k++)
		{
			for (int l = 0; l < 3; l++)
			{
				BMP_GetPixelRGBDevice(imgIn, height, width, x + k - margin, y + l - margin, &rval[m], &gval[m], &bval[m]);
				m++;
			}
		}
		BMP_SetPixelRGBDevice(imgOut, height, width, x, y, medianDevice(rval, 9), medianDevice(gval, 9), medianDevice(bval, 9));
	}
}

int checkCUDADevicesAndSetParametersToExecutionConfiguration()
{
	int devCnt;
	cudaGetDeviceCount(&devCnt);
	if (devCnt == 0)
	{
		perror("No CUDA devices available -- exiting.");
		return 1;
	}

	MAX_BLOCK_SIZE = getMaxThreadsPerBlock();

	blocks = data->size / MAX_BLOCK_SIZE;
	if (data->size % MAX_BLOCK_SIZE)
		blocks++;
}

int main(int argc, char** argv)
{
	/* Check arguments */
	if (argc != 8)
	{
		fprintf(stderr, "Usage: %s <input file> <output file>\n", argv[0]);
		return 0;
	}

	prologueCPU(argv[1]);
	data->printFlag = atoi(argv[6]);
	data->saveDataFlag = atoi(argv[7]);

	checkCUDADevicesAndSetParametersToExecutionConfiguration();

	prologueGPU();

	auto startGrayHost = std::chrono::steady_clock::now();
	cvt2GSHost(CPUBMPGrayData, data->imgWidth, data->imgHeight);
	auto endGrayHost = std::chrono::steady_clock::now();
	data->CPUGrayTime = getChronoTime(startGrayHost, endGrayHost);

	auto sMHost = std::chrono::steady_clock::now();
	medianFilterHost(CPUBMPMedianData, CPUBMPMedianDataOut, data->imgWidth, data->imgHeight);
	auto eMHost = std::chrono::steady_clock::now();
	data->CPUMedianTime = getChronoTime(sMHost, eMHost);

	auto sGD = std::chrono::steady_clock::now();
	cvt2GSDevice << <blocks, MAX_BLOCK_SIZE >> > (dBMPGrayData, data->imgWidth, data->imgHeight);
	cudaDeviceSynchronize();
	auto eGD = std::chrono::steady_clock::now();
	data->kernelGrayTime = getChronoTime(sGD, eGD);
	checkCUDAError("cvt2GSDevice");

	auto sMD = std::chrono::steady_clock::now();
	medianFilterDevice << <blocks, MAX_BLOCK_SIZE >> > (dBMPMedianData, dBMPMedianDataOut, data->imgWidth, data->imgHeight);
	cudaDeviceSynchronize();
	auto eMD = std::chrono::steady_clock::now();
	data->kernelMedianTime = getChronoTime(sMD, eMD);
	checkCUDAError("medianFilterDevice");

	epilogueGPU();
	epilogueCPU(argv[2], argv[3], argv[4], argv[5]);

	data->GPUGrayTime = data->DTHGrayTime + data->HTDGrayTime + data->kernelGrayTime;
	data->GPUMedianTime = data->DTHMedianTime + data->HTDMedianTime + data->kernelMedianTime;
	data->diffBetweenHostAndDeviceGray = data->CPUGrayTime - data->GPUGrayTime;
	data->diffBetweenHostAndDeviceMedian = data->CPUMedianTime - data->GPUMedianTime;
	
	data->printData();
	data->saveDataToFile();

	statFile.close();
	free(data);
	return 0;
}