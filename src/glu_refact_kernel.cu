#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include "/usr/local/cuda-12.2/targets/x86_64-linux/include/device_launch_parameters.h"
//#include "/usr/local/cuda-12.2/targets/x86_64-linux/include/cuda_runtime_api.h"
//#include "/usr/local/cuda-12.2/targets/x86_64-linux/include/cuda_runtime.h"
#include "lu_kernel.h"
#include "plu_kernel.h"


__host__ cudaError_t errorCheck(cudaError_t error, const char *filename, int line);
__host__ void gupdate_ls_refact(LU * lu, double * nax, double * nrhs);

__global__ void glu_refact_devkernel(int__t size, int__t *ap, int__t *ai, double *ax, \
    int__t *lp, int__t *li, double *lx, \
    int__t *up, int__t *ui, double *ux, \
    int__t *p, int__t *pinv, int__t *amdp, \
    int__t et_tlevel, int__t *et_plev, int__t *et_lists, double *wrk_buff, int *statuses, int *errs, int *issucc)
{
    int__t bid = (int__t)blockIdx.x;
    int__t loc_tid = (int__t)threadIdx.x;
    int__t i, j, jold, k, Ajnzcount;
    int__t *Ajrows, ujlen, *ujrows, row, row_new, q, *lrows, *ljrows;
    double xj, pivval;
    volatile double *Ajvalues, *ujvals, *lvals, *ljvals;
    //volatile int__t *finish;
    volatile int *wait, *breakdown;
    volatile double *work_buffer = wrk_buff + bid*size;

    // Initialize the working buffer
    for (i = loc_tid; i < size; i += blockDim.x) {
        work_buffer[i] = 0.;
    }
    __syncthreads();

    // Pipeline Mode
    int__t pipe_start = 0; // starting level in pipeline mode
    if (pipe_start < et_tlevel) {
        int__t pipe_tasks = et_plev[et_tlevel] - et_plev[pipe_start]; // number of tasks in pipeline mode
        int__t *pipe_lists = et_lists + et_plev[pipe_start];

        for (k = bid; k < pipe_tasks; k += gridDim.x) {

            j = pipe_lists[k];
            jold = (amdp != NULL) ? amdp[j]:j;

            Ajnzcount = ap[jold + 1] - ap[jold];
            Ajrows = ai + ap[jold];
            Ajvalues = ax + ap[jold];

            // numeric
            for (i = loc_tid; i < Ajnzcount; i += blockDim.x) {
                work_buffer[Ajrows[i]] = Ajvalues[i];
            }
            __syncthreads();
            

            ujrows = ui + up[j];
            ujvals = ux + up[j];
            ujlen = up[j + 1] - up[j];

            for (i = 0; i < ujlen - 1; i++) {
                row_new = ujrows[i];
                row = p[row_new];

                wait = (volatile int*)&statuses[row_new];
                while ((*wait) != DONE) {
                    for (q = 0; q < gridDim.x; q++) {
                        breakdown = (volatile int *)&errs[q];
                        if (*breakdown) {
                            return;
                        }
                    }
                }

                xj = work_buffer[row];
                lrows = li + lp[row_new];
                lvals = lx + lp[row_new];

                for (q = loc_tid; q < lp[row_new + 1]- lp[row_new]; q += blockDim.x) {
                    work_buffer[lrows[q]] -= xj*lvals[q];
                }
                __syncthreads();
            }

            // pivoting
            pivval = work_buffer[p[j]];
            if (fabs(pivval) == 0.) {
                if (loc_tid == 0) {
                    printf(" [ ERROR ] Refactorization failed because of the zero pivot.\n");
                    errs[bid] = 1;
                    *issucc = 0;
                }
                return;
            }
            __syncthreads();

            // gather L and U
            ljrows = li + lp[j];
            ljvals = lx + lp[j];
            for (i = loc_tid; i < lp[j+1] - lp[j]; i += blockDim.x) {
                row = ljrows[i];
                ljvals[i] = work_buffer[row]/pivval;
                work_buffer[row] = 0.;
            }

            for (i = loc_tid; i < ujlen; i += blockDim.x) {
                row_new = ujrows[i];
                row = p[row_new];
                ujvals[i] = work_buffer[row];
                work_buffer[row] = 0.;
            }

            __syncthreads();

            if (loc_tid == 0) {
                statuses[j] = DONE;
                //printf("Node[%d] is done status(%d) from block[%d].\n", j, statuses[j], bid);
            }
        }
    }

    /*if (bid == 0 && loc_tid == 0) {
        for (j = 0; j < size; j++) {
            printf("Lcol[%d]:", j);
            for (i = lp[j]; i < lp[j+1]; i++) {
                printf(" (%d,%9.5e)", li[i], lx[i]);
            }
            printf("\n");
        }
    }*/
}

__global__ void LUsolve_devkernel(int__t size, int__t *p, int__t *pinv, int__t *lp, int__t *li, double *lx, \
    int__t *up, int__t *ui, double *ux, \
    volatile double *b, volatile double *x)
{
    int__t bid = (int__t)blockIdx.x;
    if (gridDim.x != 1) {
        printf("[ WARNING ] Lsolve only needs one block!");
    }
    int__t loc_tid = (int__t)threadIdx.x;

    if (bid == 0) {

        int__t i, j, ljnzs, *ljrows, ujnzs, *ujrows;
        double xj, *ljvals, *ujvals;

        for (i = loc_tid; i < size; i += blockDim.x) {
            x[i] = b[i];
        }
        __syncthreads();

        for (i = loc_tid; i < size; i += blockDim.x) {
            b[i] = x[p[i]];
        }
        __syncthreads();

        for (i = loc_tid; i < size; i += blockDim.x) {
            x[i] = b[i];
        }
        __syncthreads();

        // Solving the Lower linear system...
        for (j = 0; j < size; j++) {
            ljnzs = lp[j + 1] - lp[j];
            xj = x[j];
            //__syncthreads();
            ljrows = li + lp[j];
            ljvals = lx + lp[j];
            for (i = loc_tid; i < ljnzs; i += blockDim.x) {
                x[pinv[ljrows[i]]] -= xj * ljvals[i];
            }
            __syncthreads();
        }

        // Solving the Upper linear system...
        for (j = size - 1; j >= 0; j--) {
            ujnzs = up[j + 1] - up[j];
            ujrows = ui + up[j];
            ujvals = ux + up[j];

            xj = x[j]/ujvals[ujnzs - 1];
            __syncthreads();
            if (loc_tid == 0) x[j] = xj;
            for (i = loc_tid; i < ujnzs - 1; i += blockDim.x) {
                x[ujrows[i]] -= xj * ujvals[i];
            }  
            __syncthreads();
        }
    }
}

extern "C"
{
    int glu_init(LU *lu, double *nax, double *nrhs, const int__t num_blocks, const int__t num_threads_per_block)
    {
        if (lu == NULL) {
            printf(" [ ERROR ] LU is not constructed.\n");
            return 0;
        }

        if (lu->_factflag != 1) {
            printf(" [ ERROR ] Factorization is not performed before refactorization.\n");
            return 0;
        }

        if (lu->_et == NULL) {
            printf(" [ ERROR ] Etree is not constructed.\n");
            return 0;
        }

        if (lu->_glu == NULL) {
            int dev_count = 0;
            cudaGetDeviceCount(&dev_count);
            if (dev_count <= 0) {
                printf(" [ ERROR ] CUDA device is <= 0.\n");
                return 0;
            }
            cudaSetDevice(0);
        }

        // On CPU: update the matrix and the right hand side of the equation
        gupdate_ls_refact(lu, nax, nrhs);

        int__t k, count, *lp, *li, *up, *ui;
        //double *lx, *ux;
        Glu *glu = lu->_glu;
        int__t size = lu->_mat_size;
        int__t nnzs = lu->_nnzs;
        CscMat *L = lu->_L;
        CscMat *U = lu->_U;

        if (lu->_glu == NULL)
        {
            glu = lu->_glu = (Glu *)malloc(sizeof(Glu));

            glu->_mat_size = lu->_mat_size;     // on cpu
            glu->_et_levels = lu->_et->_tlevel; // on cpu

            errorCheck(cudaMalloc((void **)&glu->_issucc, sizeof(int)), __FILE__, __LINE__);

            errorCheck(cudaMalloc((void **)&glu->_ap, (size + 1) * sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_ai, nnzs * sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_ax, nnzs * sizeof(double)), __FILE__, __LINE__);

            errorCheck(cudaMalloc((void **)&glu->_p, size * sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_pinv, size * sizeof(int__t)), __FILE__, __LINE__);
            if (lu->_amdp != NULL) errorCheck(cudaMalloc((void **)&glu->_amdp, size * sizeof(int__t)), __FILE__, __LINE__);
            else glu->_amdp = NULL;

            errorCheck(cudaMalloc((void **)&glu->_et_plev, (glu->_et_levels + 1) * sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_et_col_lists, size * sizeof(int__t)), __FILE__, __LINE__);

            if (L->_nnzs == 0) {
                L->_nnzs = 0;
                for (k = 0; k < size; k++) {
                    L->_nnzs += L->_nz_count[k];
                }
            }
            errorCheck(cudaMalloc((void **)&glu->_lp, (size + 1) * sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_li, L->_nnzs * sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_lx, L->_nnzs * sizeof(double)), __FILE__, __LINE__);

            if (U->_nnzs == 0) {
                U->_nnzs = 0;
                for (k = 0; k < size; k++) {
                    U->_nnzs += U->_nz_count[k];
                }
            }
            errorCheck(cudaMalloc((void **)&glu->_up, (size + 1) * sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_ui, U->_nnzs * sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_ux, U->_nnzs * sizeof(double)), __FILE__, __LINE__);

            errorCheck(cudaMalloc((void **)&glu->_statuses, size*sizeof(int)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_errs, num_blocks*sizeof(int)), __FILE__, __LINE__);
            //errorCheck(cudaMalloc((void **)&glu->_block_clev, num_blocks*sizeof(int__t)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_work_buff, num_blocks*size*sizeof(double)), __FILE__, __LINE__);

            errorCheck(cudaMalloc((void **)&glu->_b, size*sizeof(double)), __FILE__, __LINE__);
            errorCheck(cudaMalloc((void **)&glu->_x, size*sizeof(double)), __FILE__, __LINE__);


            cudaMemcpy(glu->_ap, lu->_ap, (size + 1)*sizeof(int__t), cudaMemcpyHostToDevice);
            cudaMemcpy(glu->_ai, lu->_ai, nnzs*sizeof(int__t), cudaMemcpyHostToDevice);
            if (lu->_amdp != NULL) cudaMemcpy(glu->_amdp, lu->_amdp, size*sizeof(int__t), cudaMemcpyHostToDevice);

            cudaMemcpy(glu->_p, lu->_p, size*sizeof(int__t), cudaMemcpyHostToDevice);
            cudaMemcpy(glu->_pinv, lu->_pinv, size*sizeof(int__t), cudaMemcpyHostToDevice);

            cudaMemcpy(glu->_et_plev, lu->_et->_plev, (glu->_et_levels + 1) * sizeof(int__t), cudaMemcpyHostToDevice);
            cudaMemcpy(glu->_et_col_lists, lu->_et->_col_lists, size*sizeof(int__t), cudaMemcpyHostToDevice);

            lp = (int__t *)calloc(size + 1, sizeof(int__t));
            up = (int__t *)calloc(size + 1, sizeof(int__t));
            for (k = 0; k < size; k++) {
                up[k + 1] = U->_nz_count[k] + up[k];
                lp[k + 1] = L->_nz_count[k] + lp[k];
            }
            cudaMemcpy(glu->_lp, lp, (size + 1)*sizeof(int__t), cudaMemcpyHostToDevice);
            cudaMemcpy(glu->_up, up, (size + 1)*sizeof(int__t), cudaMemcpyHostToDevice);
            free(lp);
            free(up);

            count = 0;
            li = (int__t *)malloc(L->_nnzs*sizeof(int__t));
            for (k = 0; k < size; k++) {
                memcpy(&li[count], &L->_rows[k][L->_nz_count[k]], L->_nz_count[k]*sizeof(int__t));
                count += L->_nz_count[k];
            }
            cudaMemcpy(glu->_li, li, L->_nnzs*sizeof(int__t), cudaMemcpyHostToDevice);
            free(li);

            count = 0;
            ui = (int__t *)malloc(U->_nnzs*sizeof(int__t));
            for (k = 0; k < size; k++) {
                memcpy(&ui[count], U->_rows[k], U->_nz_count[k]*sizeof(int__t));
                count += U->_nz_count[k];
            }
            cudaMemcpy(glu->_ui, ui, U->_nnzs*sizeof(int__t), cudaMemcpyHostToDevice);
            free(ui);
        }
        
        cudaMemcpy(glu->_ax, lu->_ax, nnzs*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(glu->_b, lu->_rhs, size*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemset(glu->_statuses, UNFINISH, size*sizeof(int));
        cudaMemset(glu->_errs, 0, num_blocks*sizeof(int));
        //cudaMemset(glu->_block_clev, 0, num_blocks*sizeof(int__t));

        cudaDeviceSynchronize();
        return 1;
    }


    int glu_refact_kernel(LU * lu, int__t num_blocks, int__t num_threads_per_block)
    {
        int__t size;
        size = lu->_mat_size;
        Glu * glu = lu->_glu;

        if (glu == NULL) {
            printf(" [ ERROR ] Please initialize the Glu at first.\n");
            return -1;
        }

        cudaEvent_t start, stop, copy_stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&copy_stop);
        
        cudaEventRecord(start, 0);
	    cudaEventSynchronize(start);

        //cudaMemset(glu->_issucc, 1, sizeof(int));
        int issucc = 1;
        cudaMemcpy(glu->_issucc, &issucc, sizeof(int), cudaMemcpyHostToDevice);

        glu_refact_devkernel<<<num_blocks, num_threads_per_block>>>(glu->_mat_size, glu->_ap, glu->_ai, glu->_ax, glu->_lp, glu->_li, glu->_lx, glu->_up, glu->_ui, glu->_ux, \
                                    glu->_p, glu->_pinv, glu->_amdp, glu->_et_levels, glu->_et_plev, glu->_et_col_lists, glu->_work_buff, glu->_statuses, glu->_errs, \
                                    glu->_issucc);


        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaDeviceSynchronize();
     
        // Check the GPU-based refactorization if is successful (On CPU)
        cudaMemcpy(&issucc, glu->_issucc, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (!issucc) {
            printf("GPU-based refactorization failed because of zero pivot.\n");
        }
        else {
            printf("GPU-based refactorization is successfull.\n");
            /*count = 0;    
            for (k = 0; k < size; k++) {
                cudaMemcpy(L->_values[k], &glu->_lx[count], L->_nz_count[k]*sizeof(double), cudaMemcpyDeviceToHost);
                count += L->_nz_count[k];
            }

            count = 0;
            for (k = 0; k < size; k++) {
                cudaMemcpy(U->_values[k], &glu->_ux[count], U->_nz_count[k]*sizeof(double), cudaMemcpyDeviceToHost);
                count += U->_nz_count[k];
            }*/

            LUsolve_devkernel<<<1, num_threads_per_block>>>(size, glu->_p, glu->_pinv, glu->_lp, glu->_li, glu->_lx, \
                glu->_up, glu->_ui, glu->_ux, \
                (volatile double*)glu->_b, (volatile double*)glu->_x);

            cudaMemcpy(lu->_x, glu->_x, size*sizeof(double), cudaMemcpyDeviceToHost);
            //cudaDeviceSynchronize();
        }

        cudaEventRecord(copy_stop, 0);
        cudaEventSynchronize(copy_stop);

        float cuda_time, copy_time;
        cudaEventElapsedTime(&cuda_time, start, stop);
        cudaEventElapsedTime(&copy_time, stop, copy_stop);
        printf("Refactorization time by CUDA: %9.5e ms, copy time: %9.5e ms\n", cuda_time, copy_time);

        cudaEventDestroy(start);
	    cudaEventDestroy(stop);
        cudaEventDestroy(copy_stop);
        return issucc;
    }
}

extern "C"
{

Glu *freeGlu(Glu *glu)
{
    if (glu == NULL) return NULL;
    cudaFree(glu->_ap);
    cudaFree(glu->_ai);
    cudaFree(glu->_ax);
    cudaFree(glu->_amdp);
    cudaFree(glu->_p);
    cudaFree(glu->_pinv);
    cudaFree(glu->_li);
    cudaFree(glu->_lp);
    cudaFree(glu->_lx);
    cudaFree(glu->_ui);
    cudaFree(glu->_up);
    cudaFree(glu->_ux);
    cudaFree(glu->_et_plev);
    cudaFree(glu->_et_col_lists);
    cudaFree(glu->_statuses);
    cudaFree(glu->_errs);
    //cudaFree(glu->_block_clev);
    cudaFree(glu->_work_buff);
    cudaFree(glu->_issucc);
    cudaFree(glu->_b);
    cudaFree(glu->_x);
    cudaDeviceSynchronize();
    free(glu);
    return NULL;
}

}

__host__ cudaError_t errorCheck(cudaError_t error, const char *filename, int line)
{
    if (error != cudaSuccess)
    {
        printf("CUDA ERROR:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line=%d\r\n",
               error, cudaGetErrorName(error), cudaGetErrorString(error), filename, line);
        return error;
    }
    return error;
}

__host__ void gupdate_ls_refact(LU * lu, double * nax, double * nrhs)
{
    int__t i, j, row;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai0 = lu->_ai0;
    double * ax = (double *)lu->_ax;
    double * b =  (double *)lu->_rhs;
    double * sc = lu->_sc;
    double * sr = lu->_sr;
    int__t * mc64pinv = lu->_mc64pinv;
    int__t * amdp = lu->_amdp;
    memcpy(ax, nax, nnzs*sizeof(double)); // update ax
    memcpy(b, nrhs, size*sizeof(double)); // update RHS

    memcpy(lu->_ax0, nax, nnzs*sizeof(double));
    memcpy(lu->_rhs0, nrhs, size*sizeof(double));
    
    // scaling
    if (lu->_scaling) {
        for (j = 0; j < size; j++) {
            for (i = ap[j]; i < ap[j + 1]; i++) {
                row = ai0[i];
                ax[i] *= sc[j] * sr[row];
            }
            b[j] *= sr[j];
        }
    }

    double * ob = NULL;
    if (lu->_rmvzero || lu->_amd == 1) ob = (double *)malloc(size*sizeof(double));

    if (lu->_rmvzero) {
        memcpy(ob, b, size*sizeof(double));
        for (i = 0; i < size; i++) {
            b[mc64pinv[i]] = ob[i];
        }
        
    }

    if (lu->_amd == 1) {
        memcpy(ob, b, size*sizeof(double));
        for (i = 0; i < size; i++) {
            b[i] = ob[amdp[i]];
        }
    }

    free(ob);
}