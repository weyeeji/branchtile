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
  float cu[32];
  float out[16];
  float maximum[2];
  float denominator[2];
  float out_1[16];
  float maximum_1[2];
  float denominator_1[2];
  float out_2[16];
  float maximum_2[2];
  float denominator_2[2];
  float out_3[16];
  float maximum_3[2];
  float denominator_3[2];
  float out_4[16];
  float maximum_4[2];
  float denominator_4[2];
  float out_5[16];
  float maximum_5[2];
  float denominator_5[2];
  float out_6[32];
  float maximum_6[4];
  float denominator_6[4];
  float scores[4];
  float previous[2];
  float maximum_clear[2];
  float factor[2];
  float subtotal[2];
  bfloat16_t probs_local_cast[2];
  float scores_1[4];
  float previous_1[2];
  float maximum_clear_1[2];
  float factor_1[2];
  float subtotal_1[2];
  bfloat16_t probs_local_cast_1[2];
  float scores_2[4];
  float previous_2[2];
  float maximum_clear_2[2];
  float factor_2[2];
  float subtotal_2[2];
  bfloat16_t probs_local_cast_2[2];
  float scores_3[4];
  float previous_3[2];
  float maximum_clear_3[2];
  float factor_3[2];
  float subtotal_3[2];
  bfloat16_t probs_local_cast_3[2];
  float scores_4[4];
  float previous_4[2];
  float maximum_clear_4[2];
  float factor_4[2];
  float subtotal_4[2];
  bfloat16_t probs_local_cast_4[2];
  float scores_5[4];
  float previous_5[2];
  float maximum_clear_5[2];
  float factor_5[2];
  float subtotal_5[2];
  bfloat16_t probs_local_cast_5[2];
  float scores_6[8];
  float previous_6[4];
  float maximum_clear_6[4];
  float factor_6[4];
  float subtotal_6[4];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 32) {
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = -CUDART_INF_F;
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)] = 0x0p+0f/*0.000000e+00*/;
  }
  int label = Labels[0];
  __syncthreads();
  if (label == 1) {
    int begin = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    int end = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    if (begin < end) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval;
        if ((i_1 < 1)) {
          condval = *(uint4*)(Q + (((((int)blockIdx.x) * 3072) + (i_1 * 1024)) + (((int)threadIdx.x) * 8)));
        } else {
          condval = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4224)) = condval;
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(out + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        if ((i_3 & 3) < 2) {
          out[i_3] = cu[(((i_3 >> 2) * 2) + (i_3 & 1))];
        }
      }
      float broadcast_var_3 = -CUDART_INF_F;
      *(float2*)(maximum + 0) = make_float2(broadcast_var_3, broadcast_var_3);
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(denominator + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        if (i_4 < 1) {
          maximum[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
          denominator[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
        }
      }
      for (int tile = 0; tile < (((end + 31) - begin) >> 5); ++tile) {
        __syncthreads();
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          int condval_1;
          if ((0 <= ((((tile * 32) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) + begin))) {
            condval_1 = Indices[((((int)blockIdx.x) * 259) + min(((((tile * 32) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) + begin), 258))];
          } else {
            condval_1 = 0;
          }
          int token = condval_1;
          bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_2;
          if ((((((((tile * 32) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 259))) {
            condval_2 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_2;
          bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_3;
          if ((((((((tile * 32) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 259))) {
            condval_3 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_3;
        }
        float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(scores + 0) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
        {
          bfloat16_t A_local[8];
          bfloat16_t B_local[4];
          __syncthreads();
          for (int ki = 0; ki < 8; ++ki) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4224)])), (&(A_local[0])));
            tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local[0])));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
          }
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 4; ++i_6) {
          float condval_4;
          if (((i_6 < 2) && ((((((tile * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin) + (i_6 & 1)) < end))) {
            condval_4 = (scores[i_6] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
          } else {
            condval_4 = -CUDART_INF_F;
          }
          scores[i_6] = condval_4;
        }
        *(float2*)(previous + 0) = *(float2*)(maximum + 0);
        __syncthreads();
        #pragma unroll
        for (int i_7 = 0; i_7 < 2; ++i_7) {
          maximum_clear[i_7] = -CUDART_INF_F;
          #pragma unroll
          for (int rv = 0; rv < 2; ++rv) {
            maximum_clear[i_7] = max(maximum_clear[i_7], scores[((i_7 * 2) + rv)]);
          }
          maximum_clear[i_7] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_7], (&(((float*)buf_dyn_shmem)[6848])));
          maximum_clear[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_7]);
          maximum[i_7] = max(maximum[i_7], maximum_clear[i_7]);
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          float condval_5;
          if ((previous[i_8] == -CUDART_INF_F)) {
            condval_5 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_5 = exp2f(((previous[i_8] - maximum[i_8]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          factor[i_8] = condval_5;
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 4; ++i_9) {
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
        for (int i_10 = 0; i_10 < 2; ++i_10) {
          subtotal[i_10] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
            subtotal[i_10] = (subtotal[i_10] + scores[((i_10 * 2) + rv_1)]);
          }
          subtotal[i_10] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal[i_10], (&(((float*)buf_dyn_shmem)[6720])));
          subtotal[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal[i_10]);
        }
        #pragma unroll
        for (int i_11 = 0; i_11 < 2; ++i_11) {
          denominator[i_11] = ((denominator[i_11] * factor[i_11]) + subtotal[i_11]);
        }
        #pragma unroll
        for (int i_12 = 0; i_12 < 16; ++i_12) {
          out[i_12] = (out[i_12] * factor[((i_12 & 3) >> 1)]);
        }
        __syncthreads();
        #pragma unroll
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          uint1 __1;
          float2 v_ = *(float2*)(scores + (i_13 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
          *(uint1*)(probs_local_cast + 0) = __1;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_13 * 256) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12416)) = *(uint1*)(probs_local_cast + 0);
        }
        {
          bfloat16_t A_local_1[8];
          bfloat16_t B_local_1[16];
          __syncthreads();
          for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12416)])), (&(A_local_1[0])));
            for (int i_14 = 0; i_14 < 2; ++i_14) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_1[(i_14 * 8)])));
            }
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
            }
          }
        }
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 32; ++i_15) {
        if (i_15 < 8) {
          cu[i_15] = out[(((i_15 >> 1) * 4) + (i_15 & 1))];
        }
      }
    }
    __syncthreads();
    if (begin < end) {
      if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          if (i_16 < 1) {
            ((float*)buf_dyn_shmem)[((i_16 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum[i_16];
            ((float*)buf_dyn_shmem)[(((i_16 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator[i_16];
          }
        }
      }
    }
  }
  if (label == 2) {
    int begin_1 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    int end_1 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    if (begin_1 < end_1) {
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_7;
        if ((i_17 < 1)) {
          condval_7 = *(uint4*)(Q + ((((((int)blockIdx.x) * 3072) + (i_17 * 1024)) + (((int)threadIdx.x) * 8)) + 1024));
        } else {
          condval_7 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_17 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4224)) = condval_7;
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 4; ++i_18) {
        float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(out_1 + (i_18 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 16; ++i_19) {
        if ((i_19 & 3) < 2) {
          out_1[i_19] = cu[((((i_19 >> 2) * 2) + (i_19 & 1)) + 8)];
        }
      }
      float broadcast_var_10 = -CUDART_INF_F;
      *(float2*)(maximum_1 + 0) = make_float2(broadcast_var_10, broadcast_var_10);
      float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(denominator_1 + 0) = make_float2(broadcast_var_11, broadcast_var_11);
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        if (i_20 < 1) {
          maximum_1[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
          denominator_1[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
        }
      }
      for (int tile_1 = 0; tile_1 < (((end_1 + 31) - begin_1) >> 5); ++tile_1) {
        __syncthreads();
        #pragma unroll
        for (int i_21 = 0; i_21 < 4; ++i_21) {
          int condval_8;
          if ((0 <= ((((tile_1 * 32) + (i_21 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1))) {
            condval_8 = Indices[((((int)blockIdx.x) * 259) + min(((((tile_1 * 32) + (i_21 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1), 258))];
          } else {
            condval_8 = 0;
          }
          int token_1 = condval_8;
          bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_9;
          if ((((((((tile_1 * 32) + (i_21 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 259))) {
            condval_9 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_9 = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_21 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_9;
          bfloat16_t broadcast_var_13 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_10;
          if ((((((((tile_1 * 32) + (i_21 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 259))) {
            condval_10 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_10 = make_uint4(__pack_nv_bfloat162(broadcast_var_13, broadcast_var_13), __pack_nv_bfloat162(broadcast_var_13, broadcast_var_13), __pack_nv_bfloat162(broadcast_var_13, broadcast_var_13), __pack_nv_bfloat162(broadcast_var_13, broadcast_var_13));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_21 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_10;
        }
        float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(scores_1 + 0) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
        {
          bfloat16_t A_local_2[8];
          bfloat16_t B_local_2[4];
          __syncthreads();
          for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4224)])), (&(A_local_2[0])));
            tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_2 >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_2[0])));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
          }
        }
        #pragma unroll
        for (int i_22 = 0; i_22 < 4; ++i_22) {
          float condval_11;
          if (((i_22 < 2) && ((((((tile_1 * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_1) + (i_22 & 1)) < end_1))) {
            condval_11 = (scores_1[i_22] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
          } else {
            condval_11 = -CUDART_INF_F;
          }
          scores_1[i_22] = condval_11;
        }
        *(float2*)(previous_1 + 0) = *(float2*)(maximum_1 + 0);
        __syncthreads();
        #pragma unroll
        for (int i_23 = 0; i_23 < 2; ++i_23) {
          maximum_clear_1[i_23] = -CUDART_INF_F;
          #pragma unroll
          for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
            maximum_clear_1[i_23] = max(maximum_clear_1[i_23], scores_1[((i_23 * 2) + rv_2)]);
          }
          maximum_clear_1[i_23] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_23], (&(((float*)buf_dyn_shmem)[6848])));
          maximum_clear_1[i_23] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_23]);
          maximum_1[i_23] = max(maximum_1[i_23], maximum_clear_1[i_23]);
        }
        #pragma unroll
        for (int i_24 = 0; i_24 < 2; ++i_24) {
          float condval_12;
          if ((previous_1[i_24] == -CUDART_INF_F)) {
            condval_12 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_12 = exp2f(((previous_1[i_24] - maximum_1[i_24]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          factor_1[i_24] = condval_12;
        }
        #pragma unroll
        for (int i_25 = 0; i_25 < 4; ++i_25) {
          float condval_13;
          if ((scores_1[i_25] == -CUDART_INF_F)) {
            condval_13 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_13 = exp2f(((scores_1[i_25] - maximum_1[(i_25 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          scores_1[i_25] = condval_13;
        }
        __syncthreads();
        #pragma unroll
        for (int i_26 = 0; i_26 < 2; ++i_26) {
          subtotal_1[i_26] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
            subtotal_1[i_26] = (subtotal_1[i_26] + scores_1[((i_26 * 2) + rv_3)]);
          }
          subtotal_1[i_26] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_26], (&(((float*)buf_dyn_shmem)[6720])));
          subtotal_1[i_26] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_26]);
        }
        #pragma unroll
        for (int i_27 = 0; i_27 < 2; ++i_27) {
          denominator_1[i_27] = ((denominator_1[i_27] * factor_1[i_27]) + subtotal_1[i_27]);
        }
        #pragma unroll
        for (int i_28 = 0; i_28 < 16; ++i_28) {
          out_1[i_28] = (out_1[i_28] * factor_1[((i_28 & 3) >> 1)]);
        }
        __syncthreads();
        #pragma unroll
        for (int i_29 = 0; i_29 < 2; ++i_29) {
          uint1 __2;
          float2 v__1 = *(float2*)(scores_1 + (i_29 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
          *(uint1*)(probs_local_cast_1 + 0) = __2;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_29 * 256) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12416)) = *(uint1*)(probs_local_cast_1 + 0);
        }
        {
          bfloat16_t A_local_3[8];
          bfloat16_t B_local_3[16];
          __syncthreads();
          for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12416)])), (&(A_local_3[0])));
            for (int i_30 = 0; i_30 < 2; ++i_30) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_30) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_3[(i_30 * 8)])));
            }
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_1 * 8) + 4)));
            }
          }
        }
      }
      #pragma unroll
      for (int i_31 = 0; i_31 < 8; ++i_31) {
        cu[(i_31 + 8)] = out_1[(((i_31 >> 1) * 4) + (i_31 & 1))];
      }
    }
    __syncthreads();
    if (begin_1 < end_1) {
      if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_32 = 0; i_32 < 2; ++i_32) {
          if (i_32 < 1) {
            ((float*)buf_dyn_shmem)[(((i_32 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 8)] = maximum_1[i_32];
            ((float*)buf_dyn_shmem)[(((i_32 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 40)] = denominator_1[i_32];
          }
        }
      }
    }
  }
  if (label == 3) {
    int begin_2 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    int end_2 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    if (begin_2 < end_2) {
      #pragma unroll
      for (int i_33 = 0; i_33 < 2; ++i_33) {
        bfloat16_t broadcast_var_15 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        int condval_15;
        if ((i_33 < 1)) {
          condval_15 = (((int)threadIdx.x) >> 4);
        } else {
          condval_15 = ((i_33 * 8) + (((int)threadIdx.x) >> 4));
        }
        uint4 condval_14;
        if ((0 <= condval_15)) {
          int64_t condval_16;
          if ((((int64_t)i_33) < (int64_t)1)) {
            condval_16 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
          } else {
            condval_16 = ((((int64_t)i_33) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
          }
          int64_t condval_17;
          if ((((int64_t)i_33) < (int64_t)1)) {
            condval_17 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
          } else {
            condval_17 = ((((int64_t)i_33) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
          }
          condval_14 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (min(condval_17, (int64_t)23) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_14 = make_uint4(__pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_33 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4224)) = condval_14;
      }
      #pragma unroll
      for (int i_34 = 0; i_34 < 4; ++i_34) {
        float broadcast_var_16 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(out_2 + (i_34 * 4)) = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
      }
      #pragma unroll
      for (int i_35 = 0; i_35 < 16; ++i_35) {
        if ((i_35 & 3) < 2) {
          out_2[i_35] = cu[(((i_35 >> 2) * 2) + (i_35 & 1))];
        }
      }
      #pragma unroll
      for (int i_36 = 0; i_36 < 8; ++i_36) {
        out_2[((((i_36 >> 1) * 4) + (i_36 & 1)) + 2)] = cu[(i_36 + 8)];
      }
      float broadcast_var_17 = -CUDART_INF_F;
      *(float2*)(maximum_2 + 0) = make_float2(broadcast_var_17, broadcast_var_17);
      float broadcast_var_18 = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(denominator_2 + 0) = make_float2(broadcast_var_18, broadcast_var_18);
      #pragma unroll
      for (int i_37 = 0; i_37 < 2; ++i_37) {
        if (i_37 < 1) {
          maximum_2[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
          denominator_2[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
        }
      }
      maximum_2[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
      denominator_2[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
      for (int tile_2 = 0; tile_2 < (((end_2 + 31) - begin_2) >> 5); ++tile_2) {
        __syncthreads();
        #pragma unroll
        for (int i_38 = 0; i_38 < 4; ++i_38) {
          int condval_18;
          if ((0 <= ((((tile_2 * 32) + (i_38 * 8)) + (((int)threadIdx.x) >> 4)) + begin_2))) {
            condval_18 = Indices[((((int)blockIdx.x) * 259) + min(((((tile_2 * 32) + (i_38 * 8)) + (((int)threadIdx.x) >> 4)) + begin_2), 258))];
          } else {
            condval_18 = 0;
          }
          int token_2 = condval_18;
          bfloat16_t broadcast_var_19 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_19;
          if ((((((((tile_2 * 32) + (i_38 * 8)) + (((int)threadIdx.x) >> 4)) + begin_2) < end_2) && (0 <= token_2)) && (token_2 < 259))) {
            condval_19 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_19 = make_uint4(__pack_nv_bfloat162(broadcast_var_19, broadcast_var_19), __pack_nv_bfloat162(broadcast_var_19, broadcast_var_19), __pack_nv_bfloat162(broadcast_var_19, broadcast_var_19), __pack_nv_bfloat162(broadcast_var_19, broadcast_var_19));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_38 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_19;
          bfloat16_t broadcast_var_20 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_20;
          if ((((((((tile_2 * 32) + (i_38 * 8)) + (((int)threadIdx.x) >> 4)) + begin_2) < end_2) && (0 <= token_2)) && (token_2 < 259))) {
            condval_20 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_20 = make_uint4(__pack_nv_bfloat162(broadcast_var_20, broadcast_var_20), __pack_nv_bfloat162(broadcast_var_20, broadcast_var_20), __pack_nv_bfloat162(broadcast_var_20, broadcast_var_20), __pack_nv_bfloat162(broadcast_var_20, broadcast_var_20));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_38 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_20;
        }
        float broadcast_var_21 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(scores_2 + 0) = make_float4(broadcast_var_21, broadcast_var_21, broadcast_var_21, broadcast_var_21);
        {
          bfloat16_t A_local_4[8];
          bfloat16_t B_local_4[4];
          __syncthreads();
          for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_4 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4224)])), (&(A_local_4[0])));
            tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_4 >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_4[0])));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_2 + 0), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + 0));
          }
        }
        #pragma unroll
        for (int i_39 = 0; i_39 < 4; ++i_39) {
          float condval_21;
          if (((((((tile_2 * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_2) + (i_39 & 1)) < end_2)) {
            condval_21 = (scores_2[i_39] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
          } else {
            condval_21 = -CUDART_INF_F;
          }
          scores_2[i_39] = condval_21;
        }
        *(float2*)(previous_2 + 0) = *(float2*)(maximum_2 + 0);
        __syncthreads();
        #pragma unroll
        for (int i_40 = 0; i_40 < 2; ++i_40) {
          maximum_clear_2[i_40] = -CUDART_INF_F;
          #pragma unroll
          for (int rv_4 = 0; rv_4 < 2; ++rv_4) {
            maximum_clear_2[i_40] = max(maximum_clear_2[i_40], scores_2[((i_40 * 2) + rv_4)]);
          }
          maximum_clear_2[i_40] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_2[i_40], (&(((float*)buf_dyn_shmem)[6848])));
          maximum_clear_2[i_40] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_2[i_40]);
          maximum_2[i_40] = max(maximum_2[i_40], maximum_clear_2[i_40]);
        }
        #pragma unroll
        for (int i_41 = 0; i_41 < 2; ++i_41) {
          float condval_22;
          if ((previous_2[i_41] == -CUDART_INF_F)) {
            condval_22 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_22 = exp2f(((previous_2[i_41] - maximum_2[i_41]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          factor_2[i_41] = condval_22;
        }
        #pragma unroll
        for (int i_42 = 0; i_42 < 4; ++i_42) {
          float condval_23;
          if ((scores_2[i_42] == -CUDART_INF_F)) {
            condval_23 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_23 = exp2f(((scores_2[i_42] - maximum_2[(i_42 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          scores_2[i_42] = condval_23;
        }
        __syncthreads();
        #pragma unroll
        for (int i_43 = 0; i_43 < 2; ++i_43) {
          subtotal_2[i_43] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_5 = 0; rv_5 < 2; ++rv_5) {
            subtotal_2[i_43] = (subtotal_2[i_43] + scores_2[((i_43 * 2) + rv_5)]);
          }
          subtotal_2[i_43] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_2[i_43], (&(((float*)buf_dyn_shmem)[6720])));
          subtotal_2[i_43] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_2[i_43]);
        }
        #pragma unroll
        for (int i_44 = 0; i_44 < 2; ++i_44) {
          denominator_2[i_44] = ((denominator_2[i_44] * factor_2[i_44]) + subtotal_2[i_44]);
        }
        #pragma unroll
        for (int i_45 = 0; i_45 < 16; ++i_45) {
          out_2[i_45] = (out_2[i_45] * factor_2[((i_45 & 3) >> 1)]);
        }
        __syncthreads();
        #pragma unroll
        for (int i_46 = 0; i_46 < 2; ++i_46) {
          uint1 __3;
          float2 v__2 = *(float2*)(scores_2 + (i_46 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
          *(uint1*)(probs_local_cast_2 + 0) = __3;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_46 * 256) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12416)) = *(uint1*)(probs_local_cast_2 + 0);
        }
        {
          bfloat16_t A_local_5[8];
          bfloat16_t B_local_5[16];
          __syncthreads();
          for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12416)])), (&(A_local_5[0])));
            for (int i_47 = 0; i_47 < 2; ++i_47) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_47) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_5[(i_47 * 8)])));
            }
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_2 + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_2 + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_2 * 8) + 4)));
            }
          }
        }
      }
      #pragma unroll
      for (int i_48 = 0; i_48 < 32; ++i_48) {
        if (i_48 < 8) {
          cu[i_48] = out_2[(((i_48 >> 1) * 4) + (i_48 & 1))];
        }
      }
      #pragma unroll
      for (int i_49 = 0; i_49 < 8; ++i_49) {
        cu[(i_49 + 8)] = out_2[((((i_49 >> 1) * 4) + (i_49 & 1)) + 2)];
      }
    }
    __syncthreads();
    if (begin_2 < end_2) {
      if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_50 = 0; i_50 < 2; ++i_50) {
          if (i_50 < 1) {
            ((float*)buf_dyn_shmem)[((i_50 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_2[i_50];
            ((float*)buf_dyn_shmem)[(((i_50 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_2[i_50];
          }
        }
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)] = maximum_2[1];
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)] = denominator_2[1];
      }
    }
  }
  if (label == 4) {
    int begin_3 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    int end_3 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    if (begin_3 < end_3) {
      #pragma unroll
      for (int i_51 = 0; i_51 < 2; ++i_51) {
        bfloat16_t broadcast_var_22 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_24;
        if ((i_51 < 1)) {
          condval_24 = *(uint4*)(Q + ((((((int)blockIdx.x) * 3072) + (i_51 * 1024)) + (((int)threadIdx.x) * 8)) + 2048));
        } else {
          condval_24 = make_uint4(__pack_nv_bfloat162(broadcast_var_22, broadcast_var_22), __pack_nv_bfloat162(broadcast_var_22, broadcast_var_22), __pack_nv_bfloat162(broadcast_var_22, broadcast_var_22), __pack_nv_bfloat162(broadcast_var_22, broadcast_var_22));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_51 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4224)) = condval_24;
      }
      #pragma unroll
      for (int i_52 = 0; i_52 < 4; ++i_52) {
        float broadcast_var_23 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(out_3 + (i_52 * 4)) = make_float4(broadcast_var_23, broadcast_var_23, broadcast_var_23, broadcast_var_23);
      }
      #pragma unroll
      for (int i_53 = 0; i_53 < 16; ++i_53) {
        if ((i_53 & 3) < 2) {
          out_3[i_53] = cu[((((i_53 >> 2) * 2) + (i_53 & 1)) + 16)];
        }
      }
      float broadcast_var_24 = -CUDART_INF_F;
      *(float2*)(maximum_3 + 0) = make_float2(broadcast_var_24, broadcast_var_24);
      float broadcast_var_25 = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(denominator_3 + 0) = make_float2(broadcast_var_25, broadcast_var_25);
      #pragma unroll
      for (int i_54 = 0; i_54 < 2; ++i_54) {
        if (i_54 < 1) {
          maximum_3[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
          denominator_3[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
        }
      }
      for (int tile_3 = 0; tile_3 < (((end_3 + 31) - begin_3) >> 5); ++tile_3) {
        __syncthreads();
        #pragma unroll
        for (int i_55 = 0; i_55 < 4; ++i_55) {
          int condval_25;
          if ((0 <= ((((tile_3 * 32) + (i_55 * 8)) + (((int)threadIdx.x) >> 4)) + begin_3))) {
            condval_25 = Indices[((((int)blockIdx.x) * 259) + min(((((tile_3 * 32) + (i_55 * 8)) + (((int)threadIdx.x) >> 4)) + begin_3), 258))];
          } else {
            condval_25 = 0;
          }
          int token_3 = condval_25;
          bfloat16_t broadcast_var_26 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_26;
          if ((((((((tile_3 * 32) + (i_55 * 8)) + (((int)threadIdx.x) >> 4)) + begin_3) < end_3) && (0 <= token_3)) && (token_3 < 259))) {
            condval_26 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_26 = make_uint4(__pack_nv_bfloat162(broadcast_var_26, broadcast_var_26), __pack_nv_bfloat162(broadcast_var_26, broadcast_var_26), __pack_nv_bfloat162(broadcast_var_26, broadcast_var_26), __pack_nv_bfloat162(broadcast_var_26, broadcast_var_26));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_55 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_26;
          bfloat16_t broadcast_var_27 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_27;
          if ((((((((tile_3 * 32) + (i_55 * 8)) + (((int)threadIdx.x) >> 4)) + begin_3) < end_3) && (0 <= token_3)) && (token_3 < 259))) {
            condval_27 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_27 = make_uint4(__pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_55 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_27;
        }
        float broadcast_var_28 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(scores_3 + 0) = make_float4(broadcast_var_28, broadcast_var_28, broadcast_var_28, broadcast_var_28);
        {
          bfloat16_t A_local_6[8];
          bfloat16_t B_local_6[4];
          __syncthreads();
          for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_6 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4224)])), (&(A_local_6[0])));
            tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_6 >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_6[0])));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_3 + 0), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + 0));
          }
        }
        #pragma unroll
        for (int i_56 = 0; i_56 < 4; ++i_56) {
          float condval_28;
          if (((i_56 < 2) && ((((((tile_3 * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_3) + (i_56 & 1)) < end_3))) {
            condval_28 = (scores_3[i_56] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
          } else {
            condval_28 = -CUDART_INF_F;
          }
          scores_3[i_56] = condval_28;
        }
        *(float2*)(previous_3 + 0) = *(float2*)(maximum_3 + 0);
        __syncthreads();
        #pragma unroll
        for (int i_57 = 0; i_57 < 2; ++i_57) {
          maximum_clear_3[i_57] = -CUDART_INF_F;
          #pragma unroll
          for (int rv_6 = 0; rv_6 < 2; ++rv_6) {
            maximum_clear_3[i_57] = max(maximum_clear_3[i_57], scores_3[((i_57 * 2) + rv_6)]);
          }
          maximum_clear_3[i_57] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_3[i_57], (&(((float*)buf_dyn_shmem)[6720])));
          maximum_clear_3[i_57] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_3[i_57]);
          maximum_3[i_57] = max(maximum_3[i_57], maximum_clear_3[i_57]);
        }
        #pragma unroll
        for (int i_58 = 0; i_58 < 2; ++i_58) {
          float condval_29;
          if ((previous_3[i_58] == -CUDART_INF_F)) {
            condval_29 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_29 = exp2f(((previous_3[i_58] - maximum_3[i_58]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          factor_3[i_58] = condval_29;
        }
        #pragma unroll
        for (int i_59 = 0; i_59 < 4; ++i_59) {
          float condval_30;
          if ((scores_3[i_59] == -CUDART_INF_F)) {
            condval_30 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_30 = exp2f(((scores_3[i_59] - maximum_3[(i_59 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          scores_3[i_59] = condval_30;
        }
        __syncthreads();
        #pragma unroll
        for (int i_60 = 0; i_60 < 2; ++i_60) {
          subtotal_3[i_60] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_7 = 0; rv_7 < 2; ++rv_7) {
            subtotal_3[i_60] = (subtotal_3[i_60] + scores_3[((i_60 * 2) + rv_7)]);
          }
          subtotal_3[i_60] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_3[i_60], (&(((float*)buf_dyn_shmem)[6848])));
          subtotal_3[i_60] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_3[i_60]);
        }
        #pragma unroll
        for (int i_61 = 0; i_61 < 2; ++i_61) {
          denominator_3[i_61] = ((denominator_3[i_61] * factor_3[i_61]) + subtotal_3[i_61]);
        }
        #pragma unroll
        for (int i_62 = 0; i_62 < 16; ++i_62) {
          out_3[i_62] = (out_3[i_62] * factor_3[((i_62 & 3) >> 1)]);
        }
        __syncthreads();
        #pragma unroll
        for (int i_63 = 0; i_63 < 2; ++i_63) {
          uint1 __4;
          float2 v__3 = *(float2*)(scores_3 + (i_63 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
          *(uint1*)(probs_local_cast_3 + 0) = __4;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_63 * 256) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12416)) = *(uint1*)(probs_local_cast_3 + 0);
        }
        {
          bfloat16_t A_local_7[8];
          bfloat16_t B_local_7[16];
          __syncthreads();
          for (int ki_7 = 0; ki_7 < 2; ++ki_7) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_7) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12416)])), (&(A_local_7[0])));
            for (int i_64 = 0; i_64 < 2; ++i_64) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_64) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_7[(i_64 * 8)])));
            }
            for (int j_3 = 0; j_3 < 2; ++j_3) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_3 + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + (j_3 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_3 + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + ((j_3 * 8) + 4)));
            }
          }
        }
      }
      #pragma unroll
      for (int i_65 = 0; i_65 < 8; ++i_65) {
        cu[(i_65 + 16)] = out_3[(((i_65 >> 1) * 4) + (i_65 & 1))];
      }
    }
    __syncthreads();
    if (begin_3 < end_3) {
      if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_66 = 0; i_66 < 2; ++i_66) {
          if (i_66 < 1) {
            ((float*)buf_dyn_shmem)[(((i_66 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 16)] = maximum_3[i_66];
            ((float*)buf_dyn_shmem)[(((i_66 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 48)] = denominator_3[i_66];
          }
        }
      }
    }
  }
  if (label == 5) {
    int begin_4 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    int end_4 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    if (begin_4 < end_4) {
      #pragma unroll
      for (int i_67 = 0; i_67 < 2; ++i_67) {
        bfloat16_t broadcast_var_29 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        int condval_32;
        if ((i_67 < 1)) {
          condval_32 = (((int)threadIdx.x) >> 4);
        } else {
          condval_32 = (((i_67 * 8) + (((int)threadIdx.x) >> 4)) + 8);
        }
        uint4 condval_31;
        if ((0 <= condval_32)) {
          int64_t condval_33;
          if ((((int64_t)i_67) < (int64_t)1)) {
            condval_33 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
          } else {
            condval_33 = (((((int64_t)i_67) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
          }
          int64_t condval_34;
          if ((((int64_t)i_67) < (int64_t)1)) {
            condval_34 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
          } else {
            condval_34 = (((((int64_t)i_67) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
          }
          condval_31 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (min(condval_34, (int64_t)23) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_31 = make_uint4(__pack_nv_bfloat162(broadcast_var_29, broadcast_var_29), __pack_nv_bfloat162(broadcast_var_29, broadcast_var_29), __pack_nv_bfloat162(broadcast_var_29, broadcast_var_29), __pack_nv_bfloat162(broadcast_var_29, broadcast_var_29));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_67 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4224)) = condval_31;
      }
      #pragma unroll
      for (int i_68 = 0; i_68 < 4; ++i_68) {
        float broadcast_var_30 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(out_4 + (i_68 * 4)) = make_float4(broadcast_var_30, broadcast_var_30, broadcast_var_30, broadcast_var_30);
      }
      #pragma unroll
      for (int i_69 = 0; i_69 < 16; ++i_69) {
        if ((i_69 & 3) < 2) {
          out_4[i_69] = cu[(((i_69 >> 2) * 2) + (i_69 & 1))];
        }
      }
      #pragma unroll
      for (int i_70 = 0; i_70 < 8; ++i_70) {
        out_4[((((i_70 >> 1) * 4) + (i_70 & 1)) + 2)] = cu[(i_70 + 16)];
      }
      float broadcast_var_31 = -CUDART_INF_F;
      *(float2*)(maximum_4 + 0) = make_float2(broadcast_var_31, broadcast_var_31);
      float broadcast_var_32 = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(denominator_4 + 0) = make_float2(broadcast_var_32, broadcast_var_32);
      #pragma unroll
      for (int i_71 = 0; i_71 < 2; ++i_71) {
        if (i_71 < 1) {
          maximum_4[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
          denominator_4[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
        }
      }
      maximum_4[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
      denominator_4[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
      for (int tile_4 = 0; tile_4 < (((end_4 + 31) - begin_4) >> 5); ++tile_4) {
        __syncthreads();
        #pragma unroll
        for (int i_72 = 0; i_72 < 4; ++i_72) {
          int condval_35;
          if ((0 <= ((((tile_4 * 32) + (i_72 * 8)) + (((int)threadIdx.x) >> 4)) + begin_4))) {
            condval_35 = Indices[((((int)blockIdx.x) * 259) + min(((((tile_4 * 32) + (i_72 * 8)) + (((int)threadIdx.x) >> 4)) + begin_4), 258))];
          } else {
            condval_35 = 0;
          }
          int token_4 = condval_35;
          bfloat16_t broadcast_var_33 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_36;
          if ((((((((tile_4 * 32) + (i_72 * 8)) + (((int)threadIdx.x) >> 4)) + begin_4) < end_4) && (0 <= token_4)) && (token_4 < 259))) {
            condval_36 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_4) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_36 = make_uint4(__pack_nv_bfloat162(broadcast_var_33, broadcast_var_33), __pack_nv_bfloat162(broadcast_var_33, broadcast_var_33), __pack_nv_bfloat162(broadcast_var_33, broadcast_var_33), __pack_nv_bfloat162(broadcast_var_33, broadcast_var_33));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_72 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_36;
          bfloat16_t broadcast_var_34 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_37;
          if ((((((((tile_4 * 32) + (i_72 * 8)) + (((int)threadIdx.x) >> 4)) + begin_4) < end_4) && (0 <= token_4)) && (token_4 < 259))) {
            condval_37 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_4) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_37 = make_uint4(__pack_nv_bfloat162(broadcast_var_34, broadcast_var_34), __pack_nv_bfloat162(broadcast_var_34, broadcast_var_34), __pack_nv_bfloat162(broadcast_var_34, broadcast_var_34), __pack_nv_bfloat162(broadcast_var_34, broadcast_var_34));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_72 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_37;
        }
        float broadcast_var_35 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(scores_4 + 0) = make_float4(broadcast_var_35, broadcast_var_35, broadcast_var_35, broadcast_var_35);
        {
          bfloat16_t A_local_8[8];
          bfloat16_t B_local_8[4];
          __syncthreads();
          for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_8 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4224)])), (&(A_local_8[0])));
            tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_8 >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_8[0])));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_4 + 0), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + 0));
          }
        }
        #pragma unroll
        for (int i_73 = 0; i_73 < 4; ++i_73) {
          float condval_38;
          if (((((((tile_4 * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_4) + (i_73 & 1)) < end_4)) {
            condval_38 = (scores_4[i_73] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
          } else {
            condval_38 = -CUDART_INF_F;
          }
          scores_4[i_73] = condval_38;
        }
        *(float2*)(previous_4 + 0) = *(float2*)(maximum_4 + 0);
        __syncthreads();
        #pragma unroll
        for (int i_74 = 0; i_74 < 2; ++i_74) {
          maximum_clear_4[i_74] = -CUDART_INF_F;
          #pragma unroll
          for (int rv_8 = 0; rv_8 < 2; ++rv_8) {
            maximum_clear_4[i_74] = max(maximum_clear_4[i_74], scores_4[((i_74 * 2) + rv_8)]);
          }
          maximum_clear_4[i_74] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_4[i_74], (&(((float*)buf_dyn_shmem)[6720])));
          maximum_clear_4[i_74] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_4[i_74]);
          maximum_4[i_74] = max(maximum_4[i_74], maximum_clear_4[i_74]);
        }
        #pragma unroll
        for (int i_75 = 0; i_75 < 2; ++i_75) {
          float condval_39;
          if ((previous_4[i_75] == -CUDART_INF_F)) {
            condval_39 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_39 = exp2f(((previous_4[i_75] - maximum_4[i_75]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          factor_4[i_75] = condval_39;
        }
        #pragma unroll
        for (int i_76 = 0; i_76 < 4; ++i_76) {
          float condval_40;
          if ((scores_4[i_76] == -CUDART_INF_F)) {
            condval_40 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_40 = exp2f(((scores_4[i_76] - maximum_4[(i_76 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          scores_4[i_76] = condval_40;
        }
        __syncthreads();
        #pragma unroll
        for (int i_77 = 0; i_77 < 2; ++i_77) {
          subtotal_4[i_77] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_9 = 0; rv_9 < 2; ++rv_9) {
            subtotal_4[i_77] = (subtotal_4[i_77] + scores_4[((i_77 * 2) + rv_9)]);
          }
          subtotal_4[i_77] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_4[i_77], (&(((float*)buf_dyn_shmem)[6848])));
          subtotal_4[i_77] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_4[i_77]);
        }
        #pragma unroll
        for (int i_78 = 0; i_78 < 2; ++i_78) {
          denominator_4[i_78] = ((denominator_4[i_78] * factor_4[i_78]) + subtotal_4[i_78]);
        }
        #pragma unroll
        for (int i_79 = 0; i_79 < 16; ++i_79) {
          out_4[i_79] = (out_4[i_79] * factor_4[((i_79 & 3) >> 1)]);
        }
        __syncthreads();
        #pragma unroll
        for (int i_80 = 0; i_80 < 2; ++i_80) {
          uint1 __5;
          float2 v__4 = *(float2*)(scores_4 + (i_80 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
          *(uint1*)(probs_local_cast_4 + 0) = __5;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_80 * 256) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12416)) = *(uint1*)(probs_local_cast_4 + 0);
        }
        {
          bfloat16_t A_local_9[8];
          bfloat16_t B_local_9[16];
          __syncthreads();
          for (int ki_9 = 0; ki_9 < 2; ++ki_9) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_9) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12416)])), (&(A_local_9[0])));
            for (int i_81 = 0; i_81 < 2; ++i_81) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_9 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_81) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_9[(i_81 * 8)])));
            }
            for (int j_4 = 0; j_4 < 2; ++j_4) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_4 + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + (j_4 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_4 + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + ((j_4 * 8) + 4)));
            }
          }
        }
      }
      #pragma unroll
      for (int i_82 = 0; i_82 < 32; ++i_82) {
        if (i_82 < 8) {
          cu[i_82] = out_4[(((i_82 >> 1) * 4) + (i_82 & 1))];
        }
      }
      #pragma unroll
      for (int i_83 = 0; i_83 < 8; ++i_83) {
        cu[(i_83 + 16)] = out_4[((((i_83 >> 1) * 4) + (i_83 & 1)) + 2)];
      }
    }
    __syncthreads();
    if (begin_4 < end_4) {
      if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_84 = 0; i_84 < 2; ++i_84) {
          if (i_84 < 1) {
            ((float*)buf_dyn_shmem)[((i_84 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_4[i_84];
            ((float*)buf_dyn_shmem)[(((i_84 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_4[i_84];
          }
        }
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_4[1];
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_4[1];
      }
    }
  }
  __syncthreads();
  if (label == 6) {
    int begin_5 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    int end_5 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    if (begin_5 < end_5) {
      #pragma unroll
      for (int i_85 = 0; i_85 < 2; ++i_85) {
        bfloat16_t broadcast_var_36 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        int condval_42;
        if ((i_85 < 1)) {
          condval_42 = ((((int)threadIdx.x) >> 4) + 8);
        } else {
          condval_42 = (((i_85 * 8) + (((int)threadIdx.x) >> 4)) + 8);
        }
        uint4 condval_41;
        if ((0 <= condval_42)) {
          int64_t condval_43;
          if ((((int64_t)i_85) < (int64_t)1)) {
            condval_43 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)8);
          } else {
            condval_43 = (((((int64_t)i_85) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
          }
          int64_t condval_44;
          if ((((int64_t)i_85) < (int64_t)1)) {
            condval_44 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)8);
          } else {
            condval_44 = (((((int64_t)i_85) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
          }
          condval_41 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (min(condval_44, (int64_t)23) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_41 = make_uint4(__pack_nv_bfloat162(broadcast_var_36, broadcast_var_36), __pack_nv_bfloat162(broadcast_var_36, broadcast_var_36), __pack_nv_bfloat162(broadcast_var_36, broadcast_var_36), __pack_nv_bfloat162(broadcast_var_36, broadcast_var_36));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_85 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4224)) = condval_41;
      }
      #pragma unroll
      for (int i_86 = 0; i_86 < 4; ++i_86) {
        float broadcast_var_37 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(out_5 + (i_86 * 4)) = make_float4(broadcast_var_37, broadcast_var_37, broadcast_var_37, broadcast_var_37);
      }
      #pragma unroll
      for (int i_87 = 0; i_87 < 16; ++i_87) {
        if ((i_87 & 3) < 2) {
          out_5[i_87] = cu[((((i_87 >> 2) * 2) + (i_87 & 1)) + 8)];
        }
      }
      #pragma unroll
      for (int i_88 = 0; i_88 < 8; ++i_88) {
        out_5[((((i_88 >> 1) * 4) + (i_88 & 1)) + 2)] = cu[(i_88 + 16)];
      }
      float broadcast_var_38 = -CUDART_INF_F;
      *(float2*)(maximum_5 + 0) = make_float2(broadcast_var_38, broadcast_var_38);
      float broadcast_var_39 = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(denominator_5 + 0) = make_float2(broadcast_var_39, broadcast_var_39);
      #pragma unroll
      for (int i_89 = 0; i_89 < 2; ++i_89) {
        if (i_89 < 1) {
          maximum_5[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
          denominator_5[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
        }
      }
      maximum_5[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
      denominator_5[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
      for (int tile_5 = 0; tile_5 < (((end_5 + 31) - begin_5) >> 5); ++tile_5) {
        __syncthreads();
        #pragma unroll
        for (int i_90 = 0; i_90 < 4; ++i_90) {
          int condval_45;
          if ((0 <= ((((tile_5 * 32) + (i_90 * 8)) + (((int)threadIdx.x) >> 4)) + begin_5))) {
            condval_45 = Indices[((((int)blockIdx.x) * 259) + min(((((tile_5 * 32) + (i_90 * 8)) + (((int)threadIdx.x) >> 4)) + begin_5), 258))];
          } else {
            condval_45 = 0;
          }
          int token_5 = condval_45;
          bfloat16_t broadcast_var_40 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_46;
          if ((((((((tile_5 * 32) + (i_90 * 8)) + (((int)threadIdx.x) >> 4)) + begin_5) < end_5) && (0 <= token_5)) && (token_5 < 259))) {
            condval_46 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_5) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_46 = make_uint4(__pack_nv_bfloat162(broadcast_var_40, broadcast_var_40), __pack_nv_bfloat162(broadcast_var_40, broadcast_var_40), __pack_nv_bfloat162(broadcast_var_40, broadcast_var_40), __pack_nv_bfloat162(broadcast_var_40, broadcast_var_40));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_90 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_46;
          bfloat16_t broadcast_var_41 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_47;
          if ((((((((tile_5 * 32) + (i_90 * 8)) + (((int)threadIdx.x) >> 4)) + begin_5) < end_5) && (0 <= token_5)) && (token_5 < 259))) {
            condval_47 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_5) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_47 = make_uint4(__pack_nv_bfloat162(broadcast_var_41, broadcast_var_41), __pack_nv_bfloat162(broadcast_var_41, broadcast_var_41), __pack_nv_bfloat162(broadcast_var_41, broadcast_var_41), __pack_nv_bfloat162(broadcast_var_41, broadcast_var_41));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_90 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_47;
        }
        float broadcast_var_42 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(scores_5 + 0) = make_float4(broadcast_var_42, broadcast_var_42, broadcast_var_42, broadcast_var_42);
        {
          bfloat16_t A_local_10[8];
          bfloat16_t B_local_10[4];
          __syncthreads();
          for (int ki_10 = 0; ki_10 < 8; ++ki_10) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_10 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4224)])), (&(A_local_10[0])));
            tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_10 >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_10[0])));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_5 + 0), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + 0));
          }
        }
        #pragma unroll
        for (int i_91 = 0; i_91 < 4; ++i_91) {
          float condval_48;
          if (((((((tile_5 * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_5) + (i_91 & 1)) < end_5)) {
            condval_48 = (scores_5[i_91] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
          } else {
            condval_48 = -CUDART_INF_F;
          }
          scores_5[i_91] = condval_48;
        }
        *(float2*)(previous_5 + 0) = *(float2*)(maximum_5 + 0);
        __syncthreads();
        #pragma unroll
        for (int i_92 = 0; i_92 < 2; ++i_92) {
          maximum_clear_5[i_92] = -CUDART_INF_F;
          #pragma unroll
          for (int rv_10 = 0; rv_10 < 2; ++rv_10) {
            maximum_clear_5[i_92] = max(maximum_clear_5[i_92], scores_5[((i_92 * 2) + rv_10)]);
          }
          maximum_clear_5[i_92] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_5[i_92], (&(((float*)buf_dyn_shmem)[6848])));
          maximum_clear_5[i_92] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_5[i_92]);
          maximum_5[i_92] = max(maximum_5[i_92], maximum_clear_5[i_92]);
        }
        #pragma unroll
        for (int i_93 = 0; i_93 < 2; ++i_93) {
          float condval_49;
          if ((previous_5[i_93] == -CUDART_INF_F)) {
            condval_49 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_49 = exp2f(((previous_5[i_93] - maximum_5[i_93]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          factor_5[i_93] = condval_49;
        }
        #pragma unroll
        for (int i_94 = 0; i_94 < 4; ++i_94) {
          float condval_50;
          if ((scores_5[i_94] == -CUDART_INF_F)) {
            condval_50 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_50 = exp2f(((scores_5[i_94] - maximum_5[(i_94 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          scores_5[i_94] = condval_50;
        }
        __syncthreads();
        #pragma unroll
        for (int i_95 = 0; i_95 < 2; ++i_95) {
          subtotal_5[i_95] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_11 = 0; rv_11 < 2; ++rv_11) {
            subtotal_5[i_95] = (subtotal_5[i_95] + scores_5[((i_95 * 2) + rv_11)]);
          }
          subtotal_5[i_95] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_5[i_95], (&(((float*)buf_dyn_shmem)[6720])));
          subtotal_5[i_95] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_5[i_95]);
        }
        #pragma unroll
        for (int i_96 = 0; i_96 < 2; ++i_96) {
          denominator_5[i_96] = ((denominator_5[i_96] * factor_5[i_96]) + subtotal_5[i_96]);
        }
        #pragma unroll
        for (int i_97 = 0; i_97 < 16; ++i_97) {
          out_5[i_97] = (out_5[i_97] * factor_5[((i_97 & 3) >> 1)]);
        }
        __syncthreads();
        #pragma unroll
        for (int i_98 = 0; i_98 < 2; ++i_98) {
          uint1 __6;
          float2 v__5 = *(float2*)(scores_5 + (i_98 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
          *(uint1*)(probs_local_cast_5 + 0) = __6;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_98 * 256) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12416)) = *(uint1*)(probs_local_cast_5 + 0);
        }
        {
          bfloat16_t A_local_11[8];
          bfloat16_t B_local_11[16];
          __syncthreads();
          for (int ki_11 = 0; ki_11 < 2; ++ki_11) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_11) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12416)])), (&(A_local_11[0])));
            for (int i_99 = 0; i_99 < 2; ++i_99) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_11 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_99) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_11[(i_99 * 8)])));
            }
            for (int j_5 = 0; j_5 < 2; ++j_5) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_5 + (j_5 * 8)), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + (j_5 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_5 + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + ((j_5 * 8) + 4)));
            }
          }
        }
      }
      #pragma unroll
      for (int i_100 = 0; i_100 < 8; ++i_100) {
        cu[(i_100 + 8)] = out_5[(((i_100 >> 1) * 4) + (i_100 & 1))];
      }
      #pragma unroll
      for (int i_101 = 0; i_101 < 8; ++i_101) {
        cu[(i_101 + 16)] = out_5[((((i_101 >> 1) * 4) + (i_101 & 1)) + 2)];
      }
    }
    __syncthreads();
    if (begin_5 < end_5) {
      if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_102 = 0; i_102 < 2; ++i_102) {
          if (i_102 < 1) {
            ((float*)buf_dyn_shmem)[(((i_102 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 8)] = maximum_5[i_102];
            ((float*)buf_dyn_shmem)[(((i_102 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 40)] = denominator_5[i_102];
          }
        }
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_5[1];
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_5[1];
      }
    }
  }
  __syncthreads();
  if (label == 7) {
    int begin_6 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    int end_6 = (((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[((((int)blockIdx.x) * 2) + 1)] - Offsets[(((int)blockIdx.x) * 2)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[(((int)blockIdx.x) * 2)]);
    if (begin_6 < end_6) {
      #pragma unroll
      for (int i_103 = 0; i_103 < 4; ++i_103) {
        bfloat16_t broadcast_var_43 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        int condval_52;
        if ((i_103 < 1)) {
          condval_52 = (((int)threadIdx.x) >> 4);
        } else {
          int condval_53;
          if ((i_103 < 2)) {
            condval_53 = ((i_103 * 8) + (((int)threadIdx.x) >> 4));
          } else {
            condval_53 = ((i_103 * 8) + (((int)threadIdx.x) >> 4));
          }
          condval_52 = condval_53;
        }
        uint4 condval_51;
        if (((i_103 < 3) && (0 <= condval_52))) {
          int64_t condval_54;
          if ((((int64_t)i_103) < (int64_t)1)) {
            condval_54 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
          } else {
            int64_t condval_55;
            if ((((int64_t)i_103) < (int64_t)2)) {
              condval_55 = ((((int64_t)i_103) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
            } else {
              condval_55 = ((((int64_t)i_103) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
            }
            condval_54 = condval_55;
          }
          int64_t condval_56;
          if ((((int64_t)i_103) < (int64_t)1)) {
            condval_56 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
          } else {
            int64_t condval_57;
            if ((((int64_t)i_103) < (int64_t)2)) {
              condval_57 = ((((int64_t)i_103) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
            } else {
              condval_57 = ((((int64_t)i_103) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
            }
            condval_56 = condval_57;
          }
          condval_51 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (min(condval_56, (int64_t)23) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_51 = make_uint4(__pack_nv_bfloat162(broadcast_var_43, broadcast_var_43), __pack_nv_bfloat162(broadcast_var_43, broadcast_var_43), __pack_nv_bfloat162(broadcast_var_43, broadcast_var_43), __pack_nv_bfloat162(broadcast_var_43, broadcast_var_43));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_103 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4224)) = condval_51;
      }
      #pragma unroll
      for (int i_104 = 0; i_104 < 8; ++i_104) {
        float broadcast_var_44 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(out_6 + (i_104 * 4)) = make_float4(broadcast_var_44, broadcast_var_44, broadcast_var_44, broadcast_var_44);
      }
      #pragma unroll
      for (int i_105 = 0; i_105 < 32; ++i_105) {
        if ((((i_105 >> 4) * 2) + ((i_105 & 3) >> 1)) < 1) {
          out_6[i_105] = cu[(((((i_105 >> 4) * 16) + (((i_105 & 3) >> 1) * 8)) + (((i_105 & 15) >> 2) * 2)) + (i_105 & 1))];
        }
      }
      #pragma unroll
      for (int i_106 = 0; i_106 < 8; ++i_106) {
        out_6[((((i_106 >> 1) * 4) + (i_106 & 1)) + 2)] = cu[(i_106 + 8)];
      }
      #pragma unroll
      for (int i_107 = 0; i_107 < 8; ++i_107) {
        out_6[((((i_107 >> 1) * 4) + (i_107 & 1)) + 16)] = cu[(i_107 + 16)];
      }
      float broadcast_var_45 = -CUDART_INF_F;
      *(float4*)(maximum_6 + 0) = make_float4(broadcast_var_45, broadcast_var_45, broadcast_var_45, broadcast_var_45);
      float broadcast_var_46 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(denominator_6 + 0) = make_float4(broadcast_var_46, broadcast_var_46, broadcast_var_46, broadcast_var_46);
      #pragma unroll
      for (int i_108 = 0; i_108 < 4; ++i_108) {
        if (i_108 < 1) {
          maximum_6[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
          denominator_6[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
        }
      }
      maximum_6[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
      denominator_6[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
      maximum_6[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
      denominator_6[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
      for (int tile_6 = 0; tile_6 < (((end_6 + 31) - begin_6) >> 5); ++tile_6) {
        __syncthreads();
        #pragma unroll
        for (int i_109 = 0; i_109 < 4; ++i_109) {
          int condval_58;
          if ((0 <= ((((tile_6 * 32) + (i_109 * 8)) + (((int)threadIdx.x) >> 4)) + begin_6))) {
            condval_58 = Indices[((((int)blockIdx.x) * 259) + min(((((tile_6 * 32) + (i_109 * 8)) + (((int)threadIdx.x) >> 4)) + begin_6), 258))];
          } else {
            condval_58 = 0;
          }
          int token_6 = condval_58;
          bfloat16_t broadcast_var_47 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_59;
          if ((((((((tile_6 * 32) + (i_109 * 8)) + (((int)threadIdx.x) >> 4)) + begin_6) < end_6) && (0 <= token_6)) && (token_6 < 259))) {
            condval_59 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_6) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_59 = make_uint4(__pack_nv_bfloat162(broadcast_var_47, broadcast_var_47), __pack_nv_bfloat162(broadcast_var_47, broadcast_var_47), __pack_nv_bfloat162(broadcast_var_47, broadcast_var_47), __pack_nv_bfloat162(broadcast_var_47, broadcast_var_47));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_109 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_59;
          bfloat16_t broadcast_var_48 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_60;
          if ((((((((tile_6 * 32) + (i_109 * 8)) + (((int)threadIdx.x) >> 4)) + begin_6) < end_6) && (0 <= token_6)) && (token_6 < 259))) {
            condval_60 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_6) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_60 = make_uint4(__pack_nv_bfloat162(broadcast_var_48, broadcast_var_48), __pack_nv_bfloat162(broadcast_var_48, broadcast_var_48), __pack_nv_bfloat162(broadcast_var_48, broadcast_var_48), __pack_nv_bfloat162(broadcast_var_48, broadcast_var_48));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_109 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_60;
        }
        #pragma unroll
        for (int i_110 = 0; i_110 < 2; ++i_110) {
          float broadcast_var_49 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(scores_6 + (i_110 * 4)) = make_float4(broadcast_var_49, broadcast_var_49, broadcast_var_49, broadcast_var_49);
        }
        {
          bfloat16_t A_local_12[16];
          bfloat16_t B_local_12[4];
          __syncthreads();
          for (int ki_12 = 0; ki_12 < 8; ++ki_12) {
            for (int i_111 = 0; i_111 < 2; ++i_111) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_12 >> 2) * 2048) + (i_111 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_12 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4224)])), (&(A_local_12[(i_111 * 8)])));
            }
            tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_12 >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_12 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_12[0])));
            for (int i_112 = 0; i_112 < 2; ++i_112) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_6 + (i_112 * 4)), reinterpret_cast<const unsigned*>(A_local_12 + (i_112 * 8)), reinterpret_cast<const unsigned*>(B_local_12 + 0));
            }
          }
        }
        #pragma unroll
        for (int i_113 = 0; i_113 < 8; ++i_113) {
          float condval_61;
          if (((i_113 < 6) && ((((((tile_6 * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_6) + (i_113 & 1)) < end_6))) {
            condval_61 = (scores_6[i_113] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
          } else {
            condval_61 = -CUDART_INF_F;
          }
          scores_6[i_113] = condval_61;
        }
        *(float4*)(previous_6 + 0) = *(float4*)(maximum_6 + 0);
        __syncthreads();
        #pragma unroll
        for (int i_114 = 0; i_114 < 4; ++i_114) {
          maximum_clear_6[i_114] = -CUDART_INF_F;
          #pragma unroll
          for (int rv_12 = 0; rv_12 < 2; ++rv_12) {
            maximum_clear_6[i_114] = max(maximum_clear_6[i_114], scores_6[((i_114 * 2) + rv_12)]);
          }
          maximum_clear_6[i_114] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_6[i_114], (&(((float*)buf_dyn_shmem)[6848])));
          maximum_clear_6[i_114] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_6[i_114]);
          maximum_6[i_114] = max(maximum_6[i_114], maximum_clear_6[i_114]);
        }
        #pragma unroll
        for (int i_115 = 0; i_115 < 4; ++i_115) {
          float condval_62;
          if ((previous_6[i_115] == -CUDART_INF_F)) {
            condval_62 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_62 = exp2f(((previous_6[i_115] - maximum_6[i_115]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          factor_6[i_115] = condval_62;
        }
        #pragma unroll
        for (int i_116 = 0; i_116 < 8; ++i_116) {
          float condval_63;
          if ((scores_6[i_116] == -CUDART_INF_F)) {
            condval_63 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_63 = exp2f(((scores_6[i_116] - maximum_6[(i_116 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          scores_6[i_116] = condval_63;
        }
        __syncthreads();
        #pragma unroll
        for (int i_117 = 0; i_117 < 4; ++i_117) {
          subtotal_6[i_117] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_13 = 0; rv_13 < 2; ++rv_13) {
            subtotal_6[i_117] = (subtotal_6[i_117] + scores_6[((i_117 * 2) + rv_13)]);
          }
          subtotal_6[i_117] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_6[i_117], (&(((float*)buf_dyn_shmem)[6720])));
          subtotal_6[i_117] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_6[i_117]);
        }
        #pragma unroll
        for (int i_118 = 0; i_118 < 4; ++i_118) {
          denominator_6[i_118] = ((denominator_6[i_118] * factor_6[i_118]) + subtotal_6[i_118]);
        }
        #pragma unroll
        for (int i_119 = 0; i_119 < 32; ++i_119) {
          out_6[i_119] = (out_6[i_119] * factor_6[(((i_119 >> 4) * 2) + ((i_119 & 3) >> 1))]);
        }
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 31) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12416)])), __pack_half2(((bfloat16_t)scores_6[0]), ((bfloat16_t)scores_6[1])), __pack_half2(((bfloat16_t)scores_6[2]), ((bfloat16_t)scores_6[3])), __pack_half2(((bfloat16_t)scores_6[4]), ((bfloat16_t)scores_6[5])), __pack_half2(((bfloat16_t)scores_6[6]), ((bfloat16_t)scores_6[7])));
        {
          bfloat16_t A_local_13[16];
          bfloat16_t B_local_13[16];
          __syncthreads();
          for (int ki_13 = 0; ki_13 < 2; ++ki_13) {
            for (int i_120 = 0; i_120 < 2; ++i_120) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_120 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12416)])), (&(A_local_13[(i_120 * 8)])));
            }
            for (int i_121 = 0; i_121 < 2; ++i_121) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_13 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_121) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_13[(i_121 * 8)])));
            }
            for (int i_122 = 0; i_122 < 2; ++i_122) {
              for (int j_6 = 0; j_6 < 2; ++j_6) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_6 + ((i_122 * 16) + (j_6 * 8))), reinterpret_cast<const unsigned*>(A_local_13 + (i_122 * 8)), reinterpret_cast<const unsigned*>(B_local_13 + (j_6 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_6 + (((i_122 * 16) + (j_6 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_13 + (i_122 * 8)), reinterpret_cast<const unsigned*>(B_local_13 + ((j_6 * 8) + 4)));
              }
            }
          }
        }
      }
      #pragma unroll
      for (int i_123 = 0; i_123 < 32; ++i_123) {
        if (i_123 < 8) {
          cu[i_123] = out_6[(((i_123 >> 1) * 4) + (i_123 & 1))];
        }
      }
      #pragma unroll
      for (int i_124 = 0; i_124 < 8; ++i_124) {
        cu[(i_124 + 8)] = out_6[((((i_124 >> 1) * 4) + (i_124 & 1)) + 2)];
      }
      #pragma unroll
      for (int i_125 = 0; i_125 < 8; ++i_125) {
        cu[(i_125 + 16)] = out_6[((((i_125 >> 1) * 4) + (i_125 & 1)) + 16)];
      }
    }
    __syncthreads();
    if (begin_6 < end_6) {
      if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_126 = 0; i_126 < 4; ++i_126) {
          if (i_126 < 1) {
            ((float*)buf_dyn_shmem)[((i_126 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_6[i_126];
            ((float*)buf_dyn_shmem)[(((i_126 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_6[i_126];
          }
        }
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)] = maximum_6[1];
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)] = denominator_6[1];
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_6[2];
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_6[2];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_127 = 0; i_127 < 16; ++i_127) {
    if (i_127 < 12) {
      *(float2*)(U + (((((((((int)blockIdx.x) * 9216) + (((int)blockIdx.y) * 3072)) + ((i_127 >> 2) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_127 & 3) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(cu + (i_127 * 2));
    }
  }
  if (((int)threadIdx.x) < 24) {
    Mout[(((((int)blockIdx.x) * 72) + (((int)blockIdx.y) * 24)) + ((int)threadIdx.x))] = ((float*)buf_dyn_shmem)[((int)threadIdx.x)];
    Lout[(((((int)blockIdx.x) * 72) + (((int)blockIdx.y) * 24)) + ((int)threadIdx.x))] = ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)];
  }
}

