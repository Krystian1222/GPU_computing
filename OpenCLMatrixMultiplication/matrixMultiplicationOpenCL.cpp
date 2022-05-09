#define _CRT_SECURE_NO_WARNINGS
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

static const int DEVICE_NAME_LEN = 128;
static char dev_name[DEVICE_NAME_LEN];

////////////////////////////////////////////////////////////////////////
float *createRandomMatrix(int rows, int cols, int maxValue)
{
	int NUMBER_OF_ELEMENTS = rows * cols;
	float* matrix = new float[NUMBER_OF_ELEMENTS];
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		matrix[i] = 0;
	}

	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		matrix[i] = rand() % maxValue + 1;
	}
	
	return matrix;
}
/////////////////////////////////////////////////////////////////
void clearOutputMatrix(float* matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		matrix[i] = 0;
	}
}
/////////////////////////////////////////////////////////////////
void printMatrix(float* matrix, int rows, int cols, char name)
{
	cout << "Matrix " << name << endl;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << matrix[i * cols + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}
////////////////////////////////////////////////////////////
void CPUMatrixMul(float* matrixA, int Arows, int Acols, 
				  float* matrixB, int Brows, int Bcols, 
				  float* matrixC, int Ccols)
{
	for (int i = 0; i < Arows; i++)
	{
		for (int j = 0; j < Bcols; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < Brows; k++)
			{
				sum += matrixA[i * Acols + k] * matrixB[k * Bcols + j];
			}
			matrixC[i * Ccols + j] = sum;
		}
	}
}
/////////////////////////////////////////////////////////////////
long long int getChronoTime(chrono::steady_clock::time_point start, chrono::steady_clock::time_point end)
{
	chrono::steady_clock::duration time_span = end - start;
	long long int milliseconds = chrono::duration_cast<chrono::milliseconds>(time_span).count();
	return milliseconds;
}
/////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	// Initialize random seed.
	srand(time(NULL));

	// List of program arguments
	/* 
	*	matrices size,
	*	max value of matrix element,
	*	print matrices switch
	*/

	int Acols = atoi(argv[1]);
	int Arows = Acols;
	int Bcols = Acols;
	int Brows = Acols;
	int maxValue = atoi(argv[2]);
	const int PRINT_MATRICES = atoi(argv[3]);

	int Ccols = Bcols;
	int Crows = Arows;

	if (Acols != Brows)
	{
		cout << "Incorrect matrices dimensions." << endl;
		exit(1);
	}

	int NUMBER_OF_OUT_MATRIX_ELEMENTS = Arows * Bcols;

	cl_int error = CL_SUCCESS;
	cl_uint platformNumber = 0;
	cl_device_id device_id;
	cl_uint deviceNumber;

	// Get platform and device info.
	error = clGetPlatformIDs(0, NULL, &platformNumber);
	if (platformNumber == 0)
	{
		cout << "No OpenCL platforms found." << endl;
	}

	// Get platform identifiers.
	cl_platform_id *platformIds = new cl_platform_id[platformNumber];
	error = clGetPlatformIDs(platformNumber, platformIds, NULL);

	// Get device count.
	error = clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, 1, &device_id, &deviceNumber);

	if (deviceNumber == 0)
	{
		cout << "No OpenCL devices found on platform " << 0 << "." << endl;
		exit(1);
	}

	// Get device name.
	error = clGetDeviceInfo(device_id, CL_DEVICE_NAME, DEVICE_NAME_LEN, dev_name, NULL);
	// cout << "device name = " << dev_name << endl;
	
	// Initialize of input and output matrices.
	float* A = createRandomMatrix(Arows, Acols, maxValue);
	float* B = createRandomMatrix(Brows, Bcols, maxValue);
	float* C = new float[NUMBER_OF_OUT_MATRIX_ELEMENTS];
	clearOutputMatrix(C, NUMBER_OF_OUT_MATRIX_ELEMENTS);

	// Create the OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error);
	if (context == NULL)
	{
		cout << "Failed to create OpenCL context." << endl;
	}

	// Create a command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, device_id, 0, &error);

	// Allocate the OpenCL buffer memory objects for source and result on the device.
	cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, Arows * Acols * sizeof(float), NULL, &error);
	cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, Brows * Bcols * sizeof(float), NULL, &error);
	cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUMBER_OF_OUT_MATRIX_ELEMENTS * sizeof(float), NULL, &error);

	// Read the OpenCL kernel in from source file.
	string str;
	string filename = "multiply.cl";
	ifstream kernelFileHandle(filename, ifstream::in);

	if (!kernelFileHandle.good())
	{
		cout << "Failed to read kernel file." << endl;
		exit(1);
	}
	kernelFileHandle.seekg(0, ios::end);
	size_t source_size = static_cast<size_t>(kernelFileHandle.tellg());

	str.reserve(static_cast<unsigned int>(kernelFileHandle.tellg()));
	kernelFileHandle.seekg(0, ios::beg);

	str.assign(istreambuf_iterator<char>(kernelFileHandle),
		istreambuf_iterator<char>());
	const char* source_str = str.c_str();

	// Create the program.
	cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_size, &error);

	if (error != CL_SUCCESS)
	{
		cout << "Failed to create program from source." << endl;
		exit(1);
	}

	// Build the program.
	error = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (error != CL_SUCCESS)
	{
		cout << "Failed to build program." << endl;
		exit(1);
	}

	// Create the kernel.
	cl_kernel kernel = clCreateKernel(program, "GPUMatrixMultiply", &error);
	if (error != CL_SUCCESS)
	{
		cout << "Failed to create kernel." << endl;
		exit(1);
	}

	// Start GPU time
	auto startGPU = chrono::steady_clock::now();

	// Set the arguments values.
	error = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&Acols);
	error = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&Bcols);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferA);
	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufferB);
	error = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufferC);

	// Asynchronous write of data to GPU device.
	clEnqueueWriteBuffer(commandQueue, bufferA, CL_FALSE, 0, Arows * Acols * sizeof(float), (void*)A, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, bufferB, CL_FALSE, 0, Brows * Bcols * sizeof(float), (void*)B, 0, NULL, NULL);

	// Launch kernel.
	size_t globalWorkSize[2] = { Ccols, Crows };
	error = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
		globalWorkSize, NULL, 0, NULL, NULL);

	if (error != CL_SUCCESS)
	{
		cout << "Failed to enqueueNDRangeKernel." << endl;
		exit(1);
	}

	// Read back results and check accumulated errors.
	error = clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, NUMBER_OF_OUT_MATRIX_ELEMENTS * sizeof(float),
		(void*)C, 0, NULL, NULL);

	// End GPU time
	auto endGPU = chrono::steady_clock::now();

	if(PRINT_MATRICES == 1)
	{
		printMatrix(A, Arows, Acols, 'A');
		printMatrix(B, Brows, Bcols, 'B');
		printMatrix(C, Crows, Ccols, 'C');
	}

	// Cleanup and free memory.
	clFlush(commandQueue);
	clFinish(commandQueue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);

	clearOutputMatrix(C, NUMBER_OF_OUT_MATRIX_ELEMENTS);

	auto startCPU = chrono::steady_clock::now();
	CPUMatrixMul(A, Arows, Acols, B, Brows, Bcols, C, Ccols);
	auto endCPU = chrono::steady_clock::now();

	if (PRINT_MATRICES == 1)
	{
		printMatrix(A, Arows, Acols, 'A');
		printMatrix(B, Brows, Bcols, 'B');
		printMatrix(C, Crows, Ccols, 'C');
	}

	cout << "Matrix size: " << Arows << endl;
	cout << "Range of elements: " << "<1, " << maxValue - 1 << ">" << endl;
	cout << "GPU elapsed time: " << getChronoTime(startGPU, endGPU) << " milliseconds." << endl;
	cout << "CPU elapsed time: " << getChronoTime(startCPU, endCPU) << " milliseconds." << endl;
	cout << endl;

	delete[] A;
	delete[] B;
	delete[] C;

	delete[] platformIds;
	return 0;
}