/*------------------------------------------------------------------------------------------------*/
/* ___::((xodCudaUtil.cu))::___

   ___::((created by eschei))___

	Purpose: CMake CUDA Accelerated Image experiments

	Revision History: 2024-04-27 - initial
*/

/*------------------------------------------------------------------------------------------------*/

#include <iostream>
#include <cuda_runtime.h>

#include "../include/xodCudaUtil.h"



void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}