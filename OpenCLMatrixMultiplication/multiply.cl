__kernel void GPUMatrixMultiply(int Acols, int Bcols, __global float *inA, __global float *inB, __global float* outC)
{
	int row = get_global_id(1);
	int col = get_global_id(0);
	float sum = 0.0f;
	for (int i = 0; i < Acols; i++)
	{
		sum += inA[row * Acols + i] * inB[i * Bcols + col];
	}
	outC[row * Bcols + col] = sum;
}