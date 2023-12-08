#ifndef GLU_REFACT_KERNEL_H_INCLUDED
#define GLU_REFACT_KERNEL_H_INCLUDED

#include "lu_kernel.h"

int glu_init(LU *lu, double *nax, double *nrhs, const int__t num_blocks, const int__t num_threads_per_block);

int glu_refact_kernel(LU * lu, int__t num_blocks, int__t num_threads_per_block);

Glu * freeGlu(Glu * glu);

#endif