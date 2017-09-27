#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<long>(const int N, const long alpha, const long* X,
    long* Y) {
	NOT_IMPLEMENTED;
}



template  <typename Dtype>
__global__ void zerout_kernel(void * mutable_gpu_data, int count, Dtype thre){
	//Dtype thre = Dtype(th);
	Dtype* data_ptr_tmp =  static_cast<Dtype*>(mutable_gpu_data);
		//  for(int i=0;i<count;i++){
		//	  if(data_ptr_tmp[i]<thre && data_ptr_tmp[i]>(-thre)){
		//		  data_ptr_tmp[i]=0;
		//	  }
		//  }
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while(tid<count){
		if(data_ptr_tmp[tid]<=thre && data_ptr_tmp[tid]>=(-thre)){
			data_ptr_tmp[tid] = 0;
		}
		tid += gridDim.x*blockDim.x;
	}
}

template  <typename Dtype>
__global__ void zerout_kernel(int count, const Dtype *x, Dtype *y, Dtype thre){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while(tid<count){
		if(x[tid]<=thre && x[tid]>=(-thre)){
			y[tid] = 0;
		}
    else {
      y[tid] = x[tid];
    }
		tid += gridDim.x*blockDim.x;
	}
}

template  <typename Dtype>
__global__ void zerout_kernel2(int count, Dtype *x, const Dtype *thresholds, int thresholds_len, Dtype weight){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while(tid<count){
    Dtype thre = thresholds[tid%thresholds_len]*weight;
		if(x[tid]<=thre && x[tid]>=(-thre)){
			x[tid] = 0;
		}
		tid += gridDim.x*blockDim.x;
	}
}

template <typename Dtype>
void caffe_gpu_zerout(void * mutable_gpu_data, const int count, Dtype th){
	zerout_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, (Dtype *)mutable_gpu_data, (Dtype *)mutable_gpu_data, th);
}

template <typename Dtype>
void caffe_gpu_zerout(int count, const Dtype *x, Dtype *y, Dtype thre){
	zerout_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, x, y, thre);
}

template <typename Dtype>
void caffe_gpu_zerout(int count, Dtype *x, const Dtype *thresholds, int thresholds_len, Dtype weight){
	zerout_kernel2<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, x, thresholds, thresholds_len, weight);
}

template void caffe_gpu_zerout<int>(void * mutable_gpu_data, const int count, int th);
template void caffe_gpu_zerout<unsigned int>(void * mutable_gpu_data, const int count, unsigned int th);
template void caffe_gpu_zerout<long>(void * mutable_gpu_data, const int count, long th);
template void caffe_gpu_zerout<unsigned long>(void * mutable_gpu_data, const int count, unsigned long th);
template void caffe_gpu_zerout<float>(void * mutable_gpu_data, const int count, float th);
template void caffe_gpu_zerout<double>(void * mutable_gpu_data, const int count, double th);

template void caffe_gpu_zerout<int>(int count, const int *x, int *y, int th);
template void caffe_gpu_zerout<unsigned int>(int count, const unsigned int *x, unsigned int *y, unsigned int th);
template void caffe_gpu_zerout<long>(int count, const long *x, long *y, long th);
template void caffe_gpu_zerout<float>(int count, const float *x, float *y, float th);
template void caffe_gpu_zerout<double>(int count, const double *x, double *y, double th);

template void caffe_gpu_zerout<float>(int count, float *x, const float *thresholds, int thresholds_len, float weight);
template void caffe_gpu_zerout<double>(int count, double *x, const double *thresholds, int thresholds_len, double weight);

/*template  <typename Dtype>
__global__ void if_zerout_fiber_kernel(
  int I, int J, int K, const Dtype *x, Dtype * y, int mode, Dtype thre)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (0 == mode) {
    int nfiber = J*K;
    while (tid < nfiber) {
      int is_zero = 1;
      for (int i = 0; i < I; ++i) {
        if (x[i*J*K + tid] > thre || x[i*J*K + tid] < -thre) {
          is_zero = 0;
          break;
        }
      }

      y[tid] = is_zero;

      tid += gridDim.x*blockDim.x;
    }
  }
  else if (1 == mode) {
    int nfiber = J*K;
    while (tid < nfiber) {
      int is_zero = 1;
      for (int i = 0; i < I; ++i) {
        if (x[i*J*K + tid] > thre || x[i*J*K + tid] < -thre) {
          is_zero = 0;
          break;
        }
      }

      y[tid] = is_zero;

      tid += gridDim.x*blockDim.x;
    }
  }
  else {
  }
}*/

template  <typename Dtype>
__global__ void shrinkage_kernel(void * mutable_gpu_data, int count, Dtype thre){
	//Dtype thre = Dtype(th);
	Dtype* data_ptr_tmp =  static_cast<Dtype*>(mutable_gpu_data);
		//  for(int i=0;i<count;i++){
		//	  if(data_ptr_tmp[i]<thre && data_ptr_tmp[i]>(-thre)){
		//		  data_ptr_tmp[i]=0;
		//	  }
		//  }
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while(tid<count){
		if(data_ptr_tmp[tid]<thre && data_ptr_tmp[tid]>(-thre)){
			data_ptr_tmp[tid] = 0;
		}
        else if (data_ptr_tmp[tid] > 0) {
            data_ptr_tmp[tid] -= thre;
        }
        else {
            data_ptr_tmp[tid] += thre;
        }
		tid += gridDim.x*blockDim.x;
	}
}

template <typename Dtype>
void caffe_gpu_shrinkage(void * mutable_gpu_data, const int count, Dtype th){
	shrinkage_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(mutable_gpu_data,  count,  th);
}

template void caffe_gpu_shrinkage<int>(void * mutable_gpu_data, const int count, int th);
template void caffe_gpu_shrinkage<unsigned int>(void * mutable_gpu_data, const int count, unsigned int th);
template void caffe_gpu_shrinkage<long>(void * mutable_gpu_data, const int count, long th);
template void caffe_gpu_shrinkage<float>(void * mutable_gpu_data, const int count, float th);
template void caffe_gpu_shrinkage<double>(void * mutable_gpu_data, const int count, double th);


template  <typename Dtype>
__global__ void if_zerout_kernel(const int n, const Dtype * x, Dtype *y, Dtype thre){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while(tid<n){
		y[tid] = (x[tid]<=thre && x[tid]>=(-thre)) ? 1 : 0;
		tid += gridDim.x*blockDim.x;
	}
}

template  <typename Dtype>
__global__ void if_zerout_kernel(const int n, const Dtype * x, Dtype *y, const Dtype *thresholds, int thresholds_len, Dtype weight){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while(tid<n){
    Dtype thre = thresholds[tid%thresholds_len]*weight;
		y[tid] = (x[tid]<=thre && x[tid]>=(-thre)) ? 1 : 0;
		tid += gridDim.x*blockDim.x;
	}
}

template <typename Dtype>
void caffe_gpu_if_zerout(const int n, const Dtype * x, Dtype *y, Dtype th){
	if_zerout_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y, th);
}

template <typename Dtype>
void caffe_gpu_if_zerout(const int n, const Dtype * x, Dtype *y, const Dtype *thresholds, int thresholds_len, Dtype weight) {
	if_zerout_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y, thresholds, thresholds_len, weight);
}

template void caffe_gpu_if_zerout<int>(const int n, const int * x, int *y, int th);
template void caffe_gpu_if_zerout<unsigned int>(const int n, const unsigned int* x, unsigned int *y, unsigned int th);
template void caffe_gpu_if_zerout<long>(const int n, const long* x, long *y, long th);

template void caffe_gpu_if_zerout<float>(const int n, const float * x, float *y, float th);
template void caffe_gpu_if_zerout<double>(const int n, const double* x, double *y, double th);

template void caffe_gpu_if_zerout<float>(const int n, const float * x, float *y, const float *thresholds, int thresholds_len, float weight);
template void caffe_gpu_if_zerout<double>(const int n, const double* x, double *y, const double *thresholds, int thresholds_len, double weight);

template  <typename Dtype>
__global__ void if_nonzerout_kernel(const int n, const Dtype * x, Dtype *y, Dtype thre){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while(tid<n){
		y[tid] = (x[tid]<=thre && x[tid]>=(-thre)) ? 0 : 1;
		tid += gridDim.x*blockDim.x;
	}
}

template <typename Dtype>
void caffe_gpu_if_nonzerout(const int n, const Dtype * x, Dtype *y, Dtype th){
	if_nonzerout_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y, th);
}

template void caffe_gpu_if_nonzerout<int>(const int n, const int * x, int *y, int th);
template void caffe_gpu_if_nonzerout<unsigned int>(const int n, const unsigned int* x, unsigned int *y, unsigned int th);
template void caffe_gpu_if_nonzerout<long>(const int n, const long* x, long*y, long th);
template void caffe_gpu_if_nonzerout<unsigned long>(const int n, const unsigned long* x, unsigned long*y, unsigned long th);

template void caffe_gpu_if_nonzerout<float>(const int n, const float * x, float *y, float th);
template void caffe_gpu_if_nonzerout<double>(const int n, const double* x, double *y, double th);

//template <>
//void caffe_gpu_zerout<int>(void * mutable_gpu_data, int count, int th){
//	zerout_kernel<<<32768,256>>>(mutable_gpu_data,  count,  th);
//}
//
//template <>
//void caffe_gpu_zerout<float>(void * mutable_gpu_data, int count, float th){
//	zerout_kernel<<<32768,256>>>(mutable_gpu_data,  count,  th);
//}
//
//template <>
//void caffe_gpu_zerout<double>(void * mutable_gpu_data, int count, double th){
//	zerout_kernel<<<32768,256>>>(mutable_gpu_data,  count,  th);
//}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<long>(const int N, const long alpha, long *X) {
   NOT_IMPLEMENTED;
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<long>(const int n, const long* x, const long* y,
    long * out) {
  NOT_IMPLEMENTED;
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<long>(const int n, const long* x, long* y) {
  NOT_IMPLEMENTED;
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <typename Dtype>
__global__ void div_checkzero_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = b[index] ? (a[index] / b[index]) : Dtype(0);
  }
}

template <typename Dtype>
__global__ void inv_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = 1 / a[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div_checkzero<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_checkzero_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div_checkzero<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	div_checkzero_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_inv<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  inv_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_inv<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  inv_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
