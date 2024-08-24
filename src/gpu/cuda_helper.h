#pragma once

#include <iostream>
#include <stdio.h>

// from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        auto error = cudaGetErrorString(result);
        std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        std::cout << error << '\n';
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}