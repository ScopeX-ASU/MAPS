#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <type_traits>
#include "cuda_runtime.h"
#include <cuComplex.h>
#include "cudss.h"

#define CUDSS_EXAMPLE_FREE       \
    do                           \
    {                            \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d);  \
        cudaFree(x_values_d);    \
        cudaFree(b_values_d);    \
    } while (0);

#define CUDA_CALL_AND_CHECK(call, msg)                                                               \
    do                                                                                               \
    {                                                                                                \
        cuda_error = call;                                                                           \
        if (cuda_error != cudaSuccess)                                                               \
        {                                                                                            \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
            CUDSS_EXAMPLE_FREE;                                                                      \
            return -1;                                                                               \
        }                                                                                            \
    } while (0);

#define CUDSS_CALL_AND_CHECK(call, status, msg)                                                                      \
    do                                                                                                               \
    {                                                                                                                \
        status = call;                                                                                               \
        if (status != CUDSS_STATUS_SUCCESS)                                                                          \
        {                                                                                                            \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE;                                                                                      \
            return -2;                                                                                               \
        }                                                                                                            \
    } while (0);

template <typename T>
int cuDSSSpsolveCUDALauncher(int *csr_offsets_d,
                             int *csr_columns_d,
                             T *csr_values_d_raw,
                             T *b_values_d_raw,
                             T *x_values_d_raw,
                             int n,
                             int nrhs,
                             int nnz,
                             int _mtype)
{
    // mtype = 0,1,2,3,4
    // CUDSS_MTYPE_GENERAL,
    // CUDSS_MTYPE_SYMMETRIC,
    // CUDSS_MTYPE_HERMITIAN,
    // CUDSS_MTYPE_SPD,
    // CUDSS_MTYPE_HPD
    using ComplexT = typename std::conditional<std::is_same<T, float>::value, cuFloatComplex, cuDoubleComplex>::type;
    constexpr auto CUDA_DTYPE = std::is_same<T, float>::value ? CUDA_C_32F : CUDA_C_64F;

    ComplexT *csr_values_d = reinterpret_cast<ComplexT *>(csr_values_d_raw);
    ComplexT *b_values_d = reinterpret_cast<ComplexT *>(b_values_d_raw);
    ComplexT *x_values_d = reinterpret_cast<ComplexT *>(x_values_d_raw);
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");
    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    // CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_DTYPE,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_DTYPE,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");
    // printf("created b and x\n");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    switch (_mtype)
    {
    case 0:
        mtype = CUDSS_MTYPE_GENERAL;
        mview = CUDSS_MVIEW_FULL;
        break;
    case 1:
        mtype = CUDSS_MTYPE_SYMMETRIC;
        mview = CUDSS_MVIEW_UPPER;
        break;
    case 2:
        mtype = CUDSS_MTYPE_HERMITIAN;
        break;
    case 3:
        mtype = CUDSS_MTYPE_SPD;
        break;
    case 4:
        mtype = CUDSS_MTYPE_HPD;
        break;
    default:
        break;
    }

    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_DTYPE, mtype, mview,
                                              base),
                         status, "cudssMatrixCreateCsr");
    // printf("created A\n");
    cudaDeviceSynchronize();
    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                                      A, x, b),
                         status, "cudssExecute for analysis");
    // printf("analysis done\n");
    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b),
                         status, "cudssExecute for factor");
    // printf("factor done\n");
    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                                      A, x, b),
                         status, "cudssExecute for solve");
    // printf("solve done\n");
    // cudaDeviceSynchronize();
    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    // cudaDeviceSynchronize();

    // printf("done\n");
    // cuComplex *x_values_h = (cuComplex*)malloc(nrhs * n * sizeof(cuComplex));
    // CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d_raw, nrhs * n * sizeof(cuComplex),
    //                     cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");
    // printf("%d %d %d\n", n, nrhs, nnz);
    // printf("%f %f\n", x_values_h[0].x, x_values_h[0].y);

    // CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(cuComplex),
    //                     cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");
    // cudaDeviceSynchronize();
    // printf("%f %f\n", x_values_h[0].x, x_values_h[0].y);
    // free(x_values_h);
    // printf("return \n");
    return 0;
}

template int cuDSSSpsolveCUDALauncher(int *csr_offsets_d,
                                      int *csr_columns_d,
                                      float *csr_values_d_raw,
                                      float *b_values_d_raw,
                                      float *x_values_d_raw,
                                      int n,
                                      int nrhs,
                                      int nnz,
                                      int _mtype);

template int cuDSSSpsolveCUDALauncher(int *csr_offsets_d,
                                      int *csr_columns_d,
                                      double *csr_values_d_raw,
                                      double *b_values_d_raw,
                                      double *x_values_d_raw,
                                      int n,
                                      int nrhs,
                                      int nnz,
                                      int _mtype);
