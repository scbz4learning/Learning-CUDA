#pragma once

#include <cuda_runtime.h>
#define RUNTIME_ERR_TYPE cudaError_t
#define RUNTIME_SUCCESS_CODE cudaSuccess
#define RUNTIME_GET_ERROR_STR cudaGetErrorString

#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    RUNTIME_ERR_TYPE err = call;                                               \
    if (err != RUNTIME_SUCCESS_CODE) {                                         \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << RUNTIME_GET_ERROR_STR(err) << "\n";                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
