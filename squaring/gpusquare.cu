/* Kernel for vector squaring */

__global__ void gpusquare(float in[], float out[], int n) 
{
   
   int i = blockDim.x * blockIdx.x + threadIdx.x;

   if (i < n)
	{ 
	out[i] = in[i] * in[i];
	}

}  