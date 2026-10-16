#include <tl_templates/cuda/instruction/mma.h>
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, const int* __restrict__ Labels, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const bfloat16_t* __restrict__ Q, float* __restrict__ U, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, const int* __restrict__ Labels, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const bfloat16_t* __restrict__ Q, float* __restrict__ U, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float out[32];
  float maximum[4];
  float denominator[4];
  float scores[8];
  float previous[4];
  float maximum_clear[4];
  float factor[4];
  float subtotal[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((i < 3)) {
      condval = *(uint4*)(Q + (((((int)blockIdx.x) * 3072) + (i * 1024)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(out + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = -CUDART_INF_F;
  *(float4*)(maximum + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(denominator + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  for (int r = 0; r < 7; ++r) {
    int begin = (((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[((((int)blockIdx.x) * 8) + r)]);
    int end = (((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[((((int)blockIdx.x) * 8) + r)]);
    int label = Labels[r];
    if (begin < end) {
      for (int tile = 0; tile < (((end + 31) - begin) >> 5); ++tile) {
        __syncthreads();
        #pragma unroll
        for (int i_2 = 0; i_2 < 4; ++i_2) {
          int condval_1;
          if ((0 <= ((((tile * 32) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + begin))) {
            condval_1 = Indices[((((int)blockIdx.x) * 259) + min(((((tile * 32) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + begin), 258))];
          } else {
            condval_1 = 0;
          }
          int token = condval_1;
          bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_2;
          if ((((((((tile * 32) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 259))) {
            condval_2 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)) = condval_2;
          bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_3;
          if ((((((((tile * 32) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 259))) {
            condval_3 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = condval_3;
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(scores + (i_3 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
        }
        {
          bfloat16_t A_local[16];
          bfloat16_t B_local[4];
          __syncthreads();
          for (int ki = 0; ki < 8; ++ki) {
            for (int i_4 = 0; i_4 < 2; ++i_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (i_4 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_4 * 8)])));
            }
            tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local[0])));
            for (int i_5 = 0; i_5 < 2; ++i_5) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (i_5 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
            }
          }
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 8; ++i_6) {
          bool valid = ((i_6 < 6) && (0 < (label & (1 << (i_6 >> 1)))));
          float condval_4;
          if ((valid && ((((((tile * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin) + (i_6 & 1)) < end))) {
            condval_4 = (scores[i_6] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
          } else {
            condval_4 = -CUDART_INF_F;
          }
          scores[i_6] = condval_4;
        }
        *(float4*)(previous + 0) = *(float4*)(maximum + 0);
        __syncthreads();
        #pragma unroll
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          maximum_clear[i_7] = -CUDART_INF_F;
          #pragma unroll
          for (int rv = 0; rv < 2; ++rv) {
            maximum_clear[i_7] = max(maximum_clear[i_7], scores[((i_7 * 2) + rv)]);
          }
          maximum_clear[i_7] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_7], (&(((float*)buf_dyn_shmem)[6784])));
          maximum_clear[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_7]);
          maximum[i_7] = max(maximum[i_7], maximum_clear[i_7]);
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 4; ++i_8) {
          float condval_5;
          if ((previous[i_8] == -CUDART_INF_F)) {
            condval_5 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_5 = exp2f(((previous[i_8] - maximum[i_8]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          factor[i_8] = condval_5;
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 8; ++i_9) {
          float condval_6;
          if ((scores[i_9] == -CUDART_INF_F)) {
            condval_6 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_6 = exp2f(((scores[i_9] - maximum[(i_9 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          scores[i_9] = condval_6;
        }
        __syncthreads();
        #pragma unroll
        for (int i_10 = 0; i_10 < 4; ++i_10) {
          subtotal[i_10] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
            subtotal[i_10] = (subtotal[i_10] + scores[((i_10 * 2) + rv_1)]);
          }
          subtotal[i_10] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal[i_10], (&(((float*)buf_dyn_shmem)[6656])));
          subtotal[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal[i_10]);
        }
        #pragma unroll
        for (int i_11 = 0; i_11 < 4; ++i_11) {
          denominator[i_11] = ((denominator[i_11] * factor[i_11]) + subtotal[i_11]);
        }
        #pragma unroll
        for (int i_12 = 0; i_12 < 32; ++i_12) {
          out[i_12] = (out[i_12] * factor[(((i_12 >> 4) * 2) + ((i_12 & 3) >> 1))]);
        }
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 31) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12288)])), __pack_half2(((bfloat16_t)scores[0]), ((bfloat16_t)scores[1])), __pack_half2(((bfloat16_t)scores[2]), ((bfloat16_t)scores[3])), __pack_half2(((bfloat16_t)scores[4]), ((bfloat16_t)scores[5])), __pack_half2(((bfloat16_t)scores[6]), ((bfloat16_t)scores[7])));
        {
          bfloat16_t A_local_1[16];
          bfloat16_t B_local_1[16];
          __syncthreads();
          for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
            for (int i_13 = 0; i_13 < 2; ++i_13) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_13 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12288)])), (&(A_local_1[(i_13 * 8)])));
            }
            for (int i_14 = 0; i_14 < 2; ++i_14) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(B_local_1[(i_14 * 8)])));
            }
            for (int i_15 = 0; i_15 < 2; ++i_15) {
              for (int j = 0; j < 2; ++j) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + ((i_15 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + (((i_15 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
              }
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    if ((((i_16 >> 3) * 2) + (i_16 & 1)) < 3) {
      *(float2*)(U + ((((((((((int)blockIdx.x) * 9216) + (((int)blockIdx.y) * 3072)) + ((i_16 >> 3) * 2048)) + ((i_16 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((i_16 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(out + (i_16 * 2));
    }
  }
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
    #pragma unroll
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      if (i_17 < 3) {
        Mout[((((((int)blockIdx.x) * 72) + (((int)blockIdx.y) * 24)) + (i_17 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = maximum[i_17];
        Lout[((((((int)blockIdx.x) * 72) + (((int)blockIdx.y) * 24)) + (i_17 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = denominator[i_17];
      }
    }
  }
}

