/*------------------------------------------------------------------------------------------------*/
/* ___::((xodCudaUtil.h))::___

   ___::((created by eschei))___

	Purpose: CMake CUDA Accelerated Image experiments

	Revision History: 2024-04-27 - initial
*/

/*------------------------------------------------------------------------------------------------*/

#ifndef __XODCUDAUTIL_H__
#define __XODCUDAUTIL_H__


#include <cuda_runtime.h>

#include "../include/xodCudaUtil.h"



void checkCudaErrors(cudaError_t err);



#endif