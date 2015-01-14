/* Kernel for vector squaring */

__global__ void threechannel(float in[], int red[], int green[], int blue[], int ret[], int num) 
{
 
	ret[threadIdx.x] = in[threadIdx.x];	  

}  
