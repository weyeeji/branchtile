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
  float out_7[16];
  float maximum_7[2];
  float denominator_7[2];
  float out_8[16];
  float maximum_8[2];
  float denominator_8[2];
  float out_9[16];
  float maximum_9[2];
  float denominator_9[2];
  float out_10[32];
  float maximum_10[4];
  float denominator_10[4];
  float out_11[16];
  float maximum_11[2];
  float denominator_11[2];
  float out_12[32];
  float maximum_12[4];
  float denominator_12[4];
  float out_13[32];
  float maximum_13[4];
  float denominator_13[4];
  float out_14[32];
  float maximum_14[4];
  float denominator_14[4];
  float scores[8];
  float previous[2];
  float maximum_clear[2];
  float factor[2];
  float subtotal[2];
  bfloat16_t probs_local_cast[2];
  float scores_1[8];
  float previous_1[2];
  float maximum_clear_1[2];
  float factor_1[2];
  float subtotal_1[2];
  bfloat16_t probs_local_cast_1[2];
  float scores_2[8];
  float previous_2[2];
  float maximum_clear_2[2];
  float factor_2[2];
  float subtotal_2[2];
  bfloat16_t probs_local_cast_2[2];
  float scores_3[8];
  float previous_3[2];
  float maximum_clear_3[2];
  float factor_3[2];
  float subtotal_3[2];
  bfloat16_t probs_local_cast_3[2];
  float scores_4[8];
  float previous_4[2];
  float maximum_clear_4[2];
  float factor_4[2];
  float subtotal_4[2];
  bfloat16_t probs_local_cast_4[2];
  float scores_5[8];
  float previous_5[2];
  float maximum_clear_5[2];
  float factor_5[2];
  float subtotal_5[2];
  bfloat16_t probs_local_cast_5[2];
  float scores_6[16];
  float previous_6[4];
  float maximum_clear_6[4];
  float factor_6[4];
  float subtotal_6[4];
  float scores_7[8];
  float previous_7[2];
  float maximum_clear_7[2];
  float factor_7[2];
  float subtotal_7[2];
  bfloat16_t probs_local_cast_6[2];
  float scores_8[8];
  float previous_8[2];
  float maximum_clear_8[2];
  float factor_8[2];
  float subtotal_8[2];
  bfloat16_t probs_local_cast_7[2];
  float scores_9[8];
  float previous_9[2];
  float maximum_clear_9[2];
  float factor_9[2];
  float subtotal_9[2];
  bfloat16_t probs_local_cast_8[2];
  float scores_10[16];
  float previous_10[4];
  float maximum_clear_10[4];
  float factor_10[4];
  float subtotal_10[4];
  float scores_11[8];
  float previous_11[2];
  float maximum_clear_11[2];
  float factor_11[2];
  float subtotal_11[2];
  bfloat16_t probs_local_cast_9[2];
  float scores_12[16];
  float previous_12[4];
  float maximum_clear_12[4];
  float factor_12[4];
  float subtotal_12[4];
  float scores_13[16];
  float previous_13[4];
  float maximum_clear_13[4];
  float factor_13[4];
  float subtotal_13[4];
  float scores_14[16];
  float previous_14[4];
  float maximum_clear_14[4];
  float factor_14[4];
  float subtotal_14[4];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 32) {
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = -CUDART_INF_F;
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)] = 0x0p+0f/*0.000000e+00*/;
  }
  __syncthreads();
  for (int r = 0; r < 15; ++r) {
    int label = Labels[r];
    if (label == 1) {
      int begin = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin < end) {
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; ++i_1) {
          bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval;
          if ((i_1 < 1)) {
            condval = *(uint4*)(Q + (((((int)blockIdx.x) * 4096) + (i_1 * 1024)) + (((int)threadIdx.x) * 8)));
          } else {
            condval = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval;
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
        for (int tile = 0; tile < (((end + 63) - begin) >> 6); ++tile) {
          __syncthreads();
          #pragma unroll
          for (int i_5 = 0; i_5 < 8; ++i_5) {
            int condval_1;
            if ((0 <= ((((tile * 64) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) + begin))) {
              condval_1 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile * 64) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) + begin), 173665))];
            } else {
              condval_1 = 0;
            }
            int token = condval_1;
            bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_2;
            if ((((((((tile * 64) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 173666))) {
              condval_2 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_2;
            bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_3;
            if ((((((((tile * 64) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 173666))) {
              condval_3 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_3;
          }
          #pragma unroll
          for (int i_6 = 0; i_6 < 2; ++i_6) {
            float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores + (i_6 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
          }
          {
            bfloat16_t A_local[8];
            bfloat16_t B_local[8];
            __syncthreads();
            for (int ki = 0; ki < 8; ++ki) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
            }
          }
          #pragma unroll
          for (int i_7 = 0; i_7 < 8; ++i_7) {
            float condval_4;
            if ((((i_7 & 3) < 2) && (((((((tile * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin) + (i_7 & 1)) < end))) {
              condval_4 = (scores[i_7] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_4 = -CUDART_INF_F;
            }
            scores[i_7] = condval_4;
          }
          *(float2*)(previous + 0) = *(float2*)(maximum + 0);
          __syncthreads();
          #pragma unroll
          for (int i_8 = 0; i_8 < 2; ++i_8) {
            maximum_clear[i_8] = -CUDART_INF_F;
            #pragma unroll
            for (int rv = 0; rv < 4; ++rv) {
              maximum_clear[i_8] = max(maximum_clear[i_8], scores[((((rv & 1) * 4) + (i_8 * 2)) + (rv >> 1))]);
            }
            maximum_clear[i_8] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_8], (&(((float*)buf_dyn_shmem)[13760])));
            maximum_clear[i_8] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_8]);
            maximum[i_8] = max(maximum[i_8], maximum_clear[i_8]);
          }
          #pragma unroll
          for (int i_9 = 0; i_9 < 2; ++i_9) {
            float condval_5;
            if ((previous[i_9] == -CUDART_INF_F)) {
              condval_5 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_5 = exp2f(((previous[i_9] - maximum[i_9]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor[i_9] = condval_5;
          }
          #pragma unroll
          for (int i_10 = 0; i_10 < 8; ++i_10) {
            float condval_6;
            if ((scores[i_10] == -CUDART_INF_F)) {
              condval_6 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_6 = exp2f(((scores[i_10] - maximum[((i_10 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores[i_10] = condval_6;
          }
          __syncthreads();
          #pragma unroll
          for (int i_11 = 0; i_11 < 2; ++i_11) {
            subtotal[i_11] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
              subtotal[i_11] = (subtotal[i_11] + scores[((((rv_1 & 1) * 4) + (i_11 * 2)) + (rv_1 >> 1))]);
            }
            subtotal[i_11] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal[i_11], (&(((float*)buf_dyn_shmem)[13376])));
            subtotal[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal[i_11]);
          }
          #pragma unroll
          for (int i_12 = 0; i_12 < 2; ++i_12) {
            denominator[i_12] = ((denominator[i_12] * factor[i_12]) + subtotal[i_12]);
          }
          #pragma unroll
          for (int i_13 = 0; i_13 < 16; ++i_13) {
            out[i_13] = (out[i_13] * factor[((i_13 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_14 = 0; i_14 < 4; ++i_14) {
            uint1 __1;
            float2 v_ = *(float2*)(scores + (i_14 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
            *(uint1*)(probs_local_cast + 0) = __1;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_14 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_14 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast + 0);
          }
          {
            bfloat16_t A_local_1[8];
            bfloat16_t B_local_1[16];
            __syncthreads();
            for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_1[0])));
              for (int i_15 = 0; i_15 < 2; ++i_15) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_15) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_1[(i_15 * 8)])));
              }
              for (int j = 0; j < 2; ++j) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_16 = 0; i_16 < 32; ++i_16) {
          if (i_16 < 8) {
            cu[i_16] = out[(((i_16 >> 1) * 4) + (i_16 & 1))];
          }
        }
      }
      __syncthreads();
      if (begin < end) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_17 = 0; i_17 < 2; ++i_17) {
            if (i_17 < 1) {
              ((float*)buf_dyn_shmem)[((i_17 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum[i_17];
              ((float*)buf_dyn_shmem)[(((i_17 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator[i_17];
            }
          }
        }
      }
    }
    if (label == 2) {
      int begin_1 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_1 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_1 < end_1) {
        #pragma unroll
        for (int i_18 = 0; i_18 < 2; ++i_18) {
          bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_7;
          if ((i_18 < 1)) {
            condval_7 = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + (i_18 * 1024)) + (((int)threadIdx.x) * 8)) + 1024));
          } else {
            condval_7 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_18 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_7;
        }
        #pragma unroll
        for (int i_19 = 0; i_19 < 4; ++i_19) {
          float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_1 + (i_19 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
        }
        #pragma unroll
        for (int i_20 = 0; i_20 < 16; ++i_20) {
          if ((i_20 & 3) < 2) {
            out_1[i_20] = cu[((((i_20 >> 2) * 2) + (i_20 & 1)) + 8)];
          }
        }
        float broadcast_var_10 = -CUDART_INF_F;
        *(float2*)(maximum_1 + 0) = make_float2(broadcast_var_10, broadcast_var_10);
        float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_1 + 0) = make_float2(broadcast_var_11, broadcast_var_11);
        #pragma unroll
        for (int i_21 = 0; i_21 < 2; ++i_21) {
          if (i_21 < 1) {
            maximum_1[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
            denominator_1[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
          }
        }
        for (int tile_1 = 0; tile_1 < (((end_1 + 63) - begin_1) >> 6); ++tile_1) {
          __syncthreads();
          #pragma unroll
          for (int i_22 = 0; i_22 < 8; ++i_22) {
            int condval_8;
            if ((0 <= ((((tile_1 * 64) + (i_22 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1))) {
              condval_8 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_1 * 64) + (i_22 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1), 173665))];
            } else {
              condval_8 = 0;
            }
            int token_1 = condval_8;
            bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_9;
            if ((((((((tile_1 * 64) + (i_22 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 173666))) {
              condval_9 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_9 = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_22 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_9;
            bfloat16_t broadcast_var_13 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_10;
            if ((((((((tile_1 * 64) + (i_22 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 173666))) {
              condval_10 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_10 = make_uint4(__pack_nv_bfloat162(broadcast_var_13, broadcast_var_13), __pack_nv_bfloat162(broadcast_var_13, broadcast_var_13), __pack_nv_bfloat162(broadcast_var_13, broadcast_var_13), __pack_nv_bfloat162(broadcast_var_13, broadcast_var_13));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_22 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_10;
          }
          #pragma unroll
          for (int i_23 = 0; i_23 < 2; ++i_23) {
            float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_1 + (i_23 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
          }
          {
            bfloat16_t A_local_2[8];
            bfloat16_t B_local_2[8];
            __syncthreads();
            for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_2[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_2[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + 4), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 4));
            }
          }
          #pragma unroll
          for (int i_24 = 0; i_24 < 8; ++i_24) {
            float condval_11;
            if ((((i_24 & 3) < 2) && (((((((tile_1 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_24 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_1) + (i_24 & 1)) < end_1))) {
              condval_11 = (scores_1[i_24] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_11 = -CUDART_INF_F;
            }
            scores_1[i_24] = condval_11;
          }
          *(float2*)(previous_1 + 0) = *(float2*)(maximum_1 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_25 = 0; i_25 < 2; ++i_25) {
            maximum_clear_1[i_25] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
              maximum_clear_1[i_25] = max(maximum_clear_1[i_25], scores_1[((((rv_2 & 1) * 4) + (i_25 * 2)) + (rv_2 >> 1))]);
            }
            maximum_clear_1[i_25] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_25], (&(((float*)buf_dyn_shmem)[11584])));
            maximum_clear_1[i_25] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_25]);
            maximum_1[i_25] = max(maximum_1[i_25], maximum_clear_1[i_25]);
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 2; ++i_26) {
            float condval_12;
            if ((previous_1[i_26] == -CUDART_INF_F)) {
              condval_12 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_12 = exp2f(((previous_1[i_26] - maximum_1[i_26]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_1[i_26] = condval_12;
          }
          #pragma unroll
          for (int i_27 = 0; i_27 < 8; ++i_27) {
            float condval_13;
            if ((scores_1[i_27] == -CUDART_INF_F)) {
              condval_13 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_13 = exp2f(((scores_1[i_27] - maximum_1[((i_27 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_1[i_27] = condval_13;
          }
          __syncthreads();
          #pragma unroll
          for (int i_28 = 0; i_28 < 2; ++i_28) {
            subtotal_1[i_28] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
              subtotal_1[i_28] = (subtotal_1[i_28] + scores_1[((((rv_3 & 1) * 4) + (i_28 * 2)) + (rv_3 >> 1))]);
            }
            subtotal_1[i_28] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_28], (&(((float*)buf_dyn_shmem)[14400])));
            subtotal_1[i_28] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_28]);
          }
          #pragma unroll
          for (int i_29 = 0; i_29 < 2; ++i_29) {
            denominator_1[i_29] = ((denominator_1[i_29] * factor_1[i_29]) + subtotal_1[i_29]);
          }
          #pragma unroll
          for (int i_30 = 0; i_30 < 16; ++i_30) {
            out_1[i_30] = (out_1[i_30] * factor_1[((i_30 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_31 = 0; i_31 < 4; ++i_31) {
            uint1 __2;
            float2 v__1 = *(float2*)(scores_1 + (i_31 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
            *(uint1*)(probs_local_cast_1 + 0) = __2;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_31 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_31 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_31 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_1 + 0);
          }
          {
            bfloat16_t A_local_3[8];
            bfloat16_t B_local_3[16];
            __syncthreads();
            for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_3[0])));
              for (int i_32 = 0; i_32 < 2; ++i_32) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_32) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_3[(i_32 * 8)])));
              }
              for (int j_1 = 0; j_1 < 2; ++j_1) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_1 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_1 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_33 = 0; i_33 < 8; ++i_33) {
          cu[(i_33 + 8)] = out_1[(((i_33 >> 1) * 4) + (i_33 & 1))];
        }
      }
      __syncthreads();
      if (begin_1 < end_1) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_34 = 0; i_34 < 2; ++i_34) {
            if (i_34 < 1) {
              ((float*)buf_dyn_shmem)[(((i_34 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 8)] = maximum_1[i_34];
              ((float*)buf_dyn_shmem)[(((i_34 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 40)] = denominator_1[i_34];
            }
          }
        }
      }
    }
    if (label == 3) {
      int begin_2 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_2 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_2 < end_2) {
        #pragma unroll
        for (int i_35 = 0; i_35 < 2; ++i_35) {
          bfloat16_t broadcast_var_15 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_15;
          if ((i_35 < 1)) {
            condval_15 = (((int)threadIdx.x) >> 4);
          } else {
            condval_15 = ((i_35 * 8) + (((int)threadIdx.x) >> 4));
          }
          uint4 condval_14;
          if ((0 <= condval_15)) {
            int64_t condval_16;
            if ((((int64_t)i_35) < (int64_t)1)) {
              condval_16 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              condval_16 = ((((int64_t)i_35) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
            }
            int64_t condval_17;
            if ((((int64_t)i_35) < (int64_t)1)) {
              condval_17 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              condval_17 = ((((int64_t)i_35) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
            }
            condval_14 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_17, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_14 = make_uint4(__pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_35 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_14;
        }
        #pragma unroll
        for (int i_36 = 0; i_36 < 4; ++i_36) {
          float broadcast_var_16 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_2 + (i_36 * 4)) = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
        }
        #pragma unroll
        for (int i_37 = 0; i_37 < 16; ++i_37) {
          if ((i_37 & 3) < 2) {
            out_2[i_37] = cu[(((i_37 >> 2) * 2) + (i_37 & 1))];
          }
        }
        #pragma unroll
        for (int i_38 = 0; i_38 < 8; ++i_38) {
          out_2[((((i_38 >> 1) * 4) + (i_38 & 1)) + 2)] = cu[(i_38 + 8)];
        }
        float broadcast_var_17 = -CUDART_INF_F;
        *(float2*)(maximum_2 + 0) = make_float2(broadcast_var_17, broadcast_var_17);
        float broadcast_var_18 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_2 + 0) = make_float2(broadcast_var_18, broadcast_var_18);
        #pragma unroll
        for (int i_39 = 0; i_39 < 2; ++i_39) {
          if (i_39 < 1) {
            maximum_2[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
            denominator_2[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
          }
        }
        maximum_2[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
        denominator_2[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
        for (int tile_2 = 0; tile_2 < (((end_2 + 63) - begin_2) >> 6); ++tile_2) {
          __syncthreads();
          #pragma unroll
          for (int i_40 = 0; i_40 < 8; ++i_40) {
            int condval_18;
            if ((0 <= ((((tile_2 * 64) + (i_40 * 8)) + (((int)threadIdx.x) >> 4)) + begin_2))) {
              condval_18 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_2 * 64) + (i_40 * 8)) + (((int)threadIdx.x) >> 4)) + begin_2), 173665))];
            } else {
              condval_18 = 0;
            }
            int token_2 = condval_18;
            bfloat16_t broadcast_var_19 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_19;
            if ((((((((tile_2 * 64) + (i_40 * 8)) + (((int)threadIdx.x) >> 4)) + begin_2) < end_2) && (0 <= token_2)) && (token_2 < 173666))) {
              condval_19 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_19 = make_uint4(__pack_nv_bfloat162(broadcast_var_19, broadcast_var_19), __pack_nv_bfloat162(broadcast_var_19, broadcast_var_19), __pack_nv_bfloat162(broadcast_var_19, broadcast_var_19), __pack_nv_bfloat162(broadcast_var_19, broadcast_var_19));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_40 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_19;
            bfloat16_t broadcast_var_20 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_20;
            if ((((((((tile_2 * 64) + (i_40 * 8)) + (((int)threadIdx.x) >> 4)) + begin_2) < end_2) && (0 <= token_2)) && (token_2 < 173666))) {
              condval_20 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_20 = make_uint4(__pack_nv_bfloat162(broadcast_var_20, broadcast_var_20), __pack_nv_bfloat162(broadcast_var_20, broadcast_var_20), __pack_nv_bfloat162(broadcast_var_20, broadcast_var_20), __pack_nv_bfloat162(broadcast_var_20, broadcast_var_20));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_40 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_20;
          }
          #pragma unroll
          for (int i_41 = 0; i_41 < 2; ++i_41) {
            float broadcast_var_21 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_2 + (i_41 * 4)) = make_float4(broadcast_var_21, broadcast_var_21, broadcast_var_21, broadcast_var_21);
          }
          {
            bfloat16_t A_local_4[8];
            bfloat16_t B_local_4[8];
            __syncthreads();
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_4 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_4[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_4 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_4[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_2 + 0), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_2 + 4), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + 4));
            }
          }
          #pragma unroll
          for (int i_42 = 0; i_42 < 8; ++i_42) {
            float condval_21;
            if ((((((((tile_2 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_42 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_2) + (i_42 & 1)) < end_2)) {
              condval_21 = (scores_2[i_42] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_21 = -CUDART_INF_F;
            }
            scores_2[i_42] = condval_21;
          }
          *(float2*)(previous_2 + 0) = *(float2*)(maximum_2 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_43 = 0; i_43 < 2; ++i_43) {
            maximum_clear_2[i_43] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_4 = 0; rv_4 < 4; ++rv_4) {
              maximum_clear_2[i_43] = max(maximum_clear_2[i_43], scores_2[((((rv_4 & 1) * 4) + (i_43 * 2)) + (rv_4 >> 1))]);
            }
            maximum_clear_2[i_43] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_2[i_43], (&(((float*)buf_dyn_shmem)[13248])));
            maximum_clear_2[i_43] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_2[i_43]);
            maximum_2[i_43] = max(maximum_2[i_43], maximum_clear_2[i_43]);
          }
          #pragma unroll
          for (int i_44 = 0; i_44 < 2; ++i_44) {
            float condval_22;
            if ((previous_2[i_44] == -CUDART_INF_F)) {
              condval_22 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_22 = exp2f(((previous_2[i_44] - maximum_2[i_44]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_2[i_44] = condval_22;
          }
          #pragma unroll
          for (int i_45 = 0; i_45 < 8; ++i_45) {
            float condval_23;
            if ((scores_2[i_45] == -CUDART_INF_F)) {
              condval_23 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_23 = exp2f(((scores_2[i_45] - maximum_2[((i_45 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_2[i_45] = condval_23;
          }
          __syncthreads();
          #pragma unroll
          for (int i_46 = 0; i_46 < 2; ++i_46) {
            subtotal_2[i_46] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_5 = 0; rv_5 < 4; ++rv_5) {
              subtotal_2[i_46] = (subtotal_2[i_46] + scores_2[((((rv_5 & 1) * 4) + (i_46 * 2)) + (rv_5 >> 1))]);
            }
            subtotal_2[i_46] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_2[i_46], (&(((float*)buf_dyn_shmem)[14528])));
            subtotal_2[i_46] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_2[i_46]);
          }
          #pragma unroll
          for (int i_47 = 0; i_47 < 2; ++i_47) {
            denominator_2[i_47] = ((denominator_2[i_47] * factor_2[i_47]) + subtotal_2[i_47]);
          }
          #pragma unroll
          for (int i_48 = 0; i_48 < 16; ++i_48) {
            out_2[i_48] = (out_2[i_48] * factor_2[((i_48 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_49 = 0; i_49 < 4; ++i_49) {
            uint1 __3;
            float2 v__2 = *(float2*)(scores_2 + (i_49 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
            *(uint1*)(probs_local_cast_2 + 0) = __3;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_49 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_49 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_49 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_2 + 0);
          }
          {
            bfloat16_t A_local_5[8];
            bfloat16_t B_local_5[16];
            __syncthreads();
            for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_5[0])));
              for (int i_50 = 0; i_50 < 2; ++i_50) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_50) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_5[(i_50 * 8)])));
              }
              for (int j_2 = 0; j_2 < 2; ++j_2) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_2 + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_2 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_2 + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_2 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_51 = 0; i_51 < 32; ++i_51) {
          if (i_51 < 8) {
            cu[i_51] = out_2[(((i_51 >> 1) * 4) + (i_51 & 1))];
          }
        }
        #pragma unroll
        for (int i_52 = 0; i_52 < 8; ++i_52) {
          cu[(i_52 + 8)] = out_2[((((i_52 >> 1) * 4) + (i_52 & 1)) + 2)];
        }
      }
      __syncthreads();
      if (begin_2 < end_2) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_53 = 0; i_53 < 2; ++i_53) {
            if (i_53 < 1) {
              ((float*)buf_dyn_shmem)[((i_53 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_2[i_53];
              ((float*)buf_dyn_shmem)[(((i_53 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_2[i_53];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)] = maximum_2[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)] = denominator_2[1];
        }
      }
    }
    if (label == 4) {
      int begin_3 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_3 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_3 < end_3) {
        #pragma unroll
        for (int i_54 = 0; i_54 < 2; ++i_54) {
          bfloat16_t broadcast_var_22 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_24;
          if ((i_54 < 1)) {
            condval_24 = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + (i_54 * 1024)) + (((int)threadIdx.x) * 8)) + 2048));
          } else {
            condval_24 = make_uint4(__pack_nv_bfloat162(broadcast_var_22, broadcast_var_22), __pack_nv_bfloat162(broadcast_var_22, broadcast_var_22), __pack_nv_bfloat162(broadcast_var_22, broadcast_var_22), __pack_nv_bfloat162(broadcast_var_22, broadcast_var_22));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_54 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_24;
        }
        #pragma unroll
        for (int i_55 = 0; i_55 < 4; ++i_55) {
          float broadcast_var_23 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_3 + (i_55 * 4)) = make_float4(broadcast_var_23, broadcast_var_23, broadcast_var_23, broadcast_var_23);
        }
        #pragma unroll
        for (int i_56 = 0; i_56 < 16; ++i_56) {
          if ((i_56 & 3) < 2) {
            out_3[i_56] = cu[((((i_56 >> 2) * 2) + (i_56 & 1)) + 16)];
          }
        }
        float broadcast_var_24 = -CUDART_INF_F;
        *(float2*)(maximum_3 + 0) = make_float2(broadcast_var_24, broadcast_var_24);
        float broadcast_var_25 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_3 + 0) = make_float2(broadcast_var_25, broadcast_var_25);
        #pragma unroll
        for (int i_57 = 0; i_57 < 2; ++i_57) {
          if (i_57 < 1) {
            maximum_3[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
            denominator_3[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
          }
        }
        for (int tile_3 = 0; tile_3 < (((end_3 + 63) - begin_3) >> 6); ++tile_3) {
          __syncthreads();
          #pragma unroll
          for (int i_58 = 0; i_58 < 8; ++i_58) {
            int condval_25;
            if ((0 <= ((((tile_3 * 64) + (i_58 * 8)) + (((int)threadIdx.x) >> 4)) + begin_3))) {
              condval_25 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_3 * 64) + (i_58 * 8)) + (((int)threadIdx.x) >> 4)) + begin_3), 173665))];
            } else {
              condval_25 = 0;
            }
            int token_3 = condval_25;
            bfloat16_t broadcast_var_26 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_26;
            if ((((((((tile_3 * 64) + (i_58 * 8)) + (((int)threadIdx.x) >> 4)) + begin_3) < end_3) && (0 <= token_3)) && (token_3 < 173666))) {
              condval_26 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_26 = make_uint4(__pack_nv_bfloat162(broadcast_var_26, broadcast_var_26), __pack_nv_bfloat162(broadcast_var_26, broadcast_var_26), __pack_nv_bfloat162(broadcast_var_26, broadcast_var_26), __pack_nv_bfloat162(broadcast_var_26, broadcast_var_26));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_58 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_26;
            bfloat16_t broadcast_var_27 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_27;
            if ((((((((tile_3 * 64) + (i_58 * 8)) + (((int)threadIdx.x) >> 4)) + begin_3) < end_3) && (0 <= token_3)) && (token_3 < 173666))) {
              condval_27 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_27 = make_uint4(__pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_58 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_27;
          }
          #pragma unroll
          for (int i_59 = 0; i_59 < 2; ++i_59) {
            float broadcast_var_28 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_3 + (i_59 * 4)) = make_float4(broadcast_var_28, broadcast_var_28, broadcast_var_28, broadcast_var_28);
          }
          {
            bfloat16_t A_local_6[8];
            bfloat16_t B_local_6[8];
            __syncthreads();
            for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_6 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_6[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_6 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_6[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_3 + 0), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_3 + 4), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + 4));
            }
          }
          #pragma unroll
          for (int i_60 = 0; i_60 < 8; ++i_60) {
            float condval_28;
            if ((((i_60 & 3) < 2) && (((((((tile_3 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_60 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_3) + (i_60 & 1)) < end_3))) {
              condval_28 = (scores_3[i_60] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_28 = -CUDART_INF_F;
            }
            scores_3[i_60] = condval_28;
          }
          *(float2*)(previous_3 + 0) = *(float2*)(maximum_3 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_61 = 0; i_61 < 2; ++i_61) {
            maximum_clear_3[i_61] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_6 = 0; rv_6 < 4; ++rv_6) {
              maximum_clear_3[i_61] = max(maximum_clear_3[i_61], scores_3[((((rv_6 & 1) * 4) + (i_61 * 2)) + (rv_6 >> 1))]);
            }
            maximum_clear_3[i_61] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_3[i_61], (&(((float*)buf_dyn_shmem)[12352])));
            maximum_clear_3[i_61] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_3[i_61]);
            maximum_3[i_61] = max(maximum_3[i_61], maximum_clear_3[i_61]);
          }
          #pragma unroll
          for (int i_62 = 0; i_62 < 2; ++i_62) {
            float condval_29;
            if ((previous_3[i_62] == -CUDART_INF_F)) {
              condval_29 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_29 = exp2f(((previous_3[i_62] - maximum_3[i_62]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_3[i_62] = condval_29;
          }
          #pragma unroll
          for (int i_63 = 0; i_63 < 8; ++i_63) {
            float condval_30;
            if ((scores_3[i_63] == -CUDART_INF_F)) {
              condval_30 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_30 = exp2f(((scores_3[i_63] - maximum_3[((i_63 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_3[i_63] = condval_30;
          }
          __syncthreads();
          #pragma unroll
          for (int i_64 = 0; i_64 < 2; ++i_64) {
            subtotal_3[i_64] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_7 = 0; rv_7 < 4; ++rv_7) {
              subtotal_3[i_64] = (subtotal_3[i_64] + scores_3[((((rv_7 & 1) * 4) + (i_64 * 2)) + (rv_7 >> 1))]);
            }
            subtotal_3[i_64] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_3[i_64], (&(((float*)buf_dyn_shmem)[12992])));
            subtotal_3[i_64] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_3[i_64]);
          }
          #pragma unroll
          for (int i_65 = 0; i_65 < 2; ++i_65) {
            denominator_3[i_65] = ((denominator_3[i_65] * factor_3[i_65]) + subtotal_3[i_65]);
          }
          #pragma unroll
          for (int i_66 = 0; i_66 < 16; ++i_66) {
            out_3[i_66] = (out_3[i_66] * factor_3[((i_66 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_67 = 0; i_67 < 4; ++i_67) {
            uint1 __4;
            float2 v__3 = *(float2*)(scores_3 + (i_67 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
            *(uint1*)(probs_local_cast_3 + 0) = __4;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_67 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_67 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_67 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_3 + 0);
          }
          {
            bfloat16_t A_local_7[8];
            bfloat16_t B_local_7[16];
            __syncthreads();
            for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_7[0])));
              for (int i_68 = 0; i_68 < 2; ++i_68) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_68) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_7[(i_68 * 8)])));
              }
              for (int j_3 = 0; j_3 < 2; ++j_3) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_3 + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + (j_3 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_3 + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + ((j_3 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_69 = 0; i_69 < 8; ++i_69) {
          cu[(i_69 + 16)] = out_3[(((i_69 >> 1) * 4) + (i_69 & 1))];
        }
      }
      __syncthreads();
      if (begin_3 < end_3) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_70 = 0; i_70 < 2; ++i_70) {
            if (i_70 < 1) {
              ((float*)buf_dyn_shmem)[(((i_70 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 16)] = maximum_3[i_70];
              ((float*)buf_dyn_shmem)[(((i_70 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 48)] = denominator_3[i_70];
            }
          }
        }
      }
    }
    if (label == 5) {
      int begin_4 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_4 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_4 < end_4) {
        #pragma unroll
        for (int i_71 = 0; i_71 < 2; ++i_71) {
          bfloat16_t broadcast_var_29 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_32;
          if ((i_71 < 1)) {
            condval_32 = (((int)threadIdx.x) >> 4);
          } else {
            condval_32 = (((i_71 * 8) + (((int)threadIdx.x) >> 4)) + 8);
          }
          uint4 condval_31;
          if ((0 <= condval_32)) {
            int64_t condval_33;
            if ((((int64_t)i_71) < (int64_t)1)) {
              condval_33 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              condval_33 = (((((int64_t)i_71) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
            }
            int64_t condval_34;
            if ((((int64_t)i_71) < (int64_t)1)) {
              condval_34 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              condval_34 = (((((int64_t)i_71) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
            }
            condval_31 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_34, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_31 = make_uint4(__pack_nv_bfloat162(broadcast_var_29, broadcast_var_29), __pack_nv_bfloat162(broadcast_var_29, broadcast_var_29), __pack_nv_bfloat162(broadcast_var_29, broadcast_var_29), __pack_nv_bfloat162(broadcast_var_29, broadcast_var_29));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_71 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_31;
        }
        #pragma unroll
        for (int i_72 = 0; i_72 < 4; ++i_72) {
          float broadcast_var_30 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_4 + (i_72 * 4)) = make_float4(broadcast_var_30, broadcast_var_30, broadcast_var_30, broadcast_var_30);
        }
        #pragma unroll
        for (int i_73 = 0; i_73 < 16; ++i_73) {
          if ((i_73 & 3) < 2) {
            out_4[i_73] = cu[(((i_73 >> 2) * 2) + (i_73 & 1))];
          }
        }
        #pragma unroll
        for (int i_74 = 0; i_74 < 8; ++i_74) {
          out_4[((((i_74 >> 1) * 4) + (i_74 & 1)) + 2)] = cu[(i_74 + 16)];
        }
        float broadcast_var_31 = -CUDART_INF_F;
        *(float2*)(maximum_4 + 0) = make_float2(broadcast_var_31, broadcast_var_31);
        float broadcast_var_32 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_4 + 0) = make_float2(broadcast_var_32, broadcast_var_32);
        #pragma unroll
        for (int i_75 = 0; i_75 < 2; ++i_75) {
          if (i_75 < 1) {
            maximum_4[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
            denominator_4[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
          }
        }
        maximum_4[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
        denominator_4[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
        for (int tile_4 = 0; tile_4 < (((end_4 + 63) - begin_4) >> 6); ++tile_4) {
          __syncthreads();
          #pragma unroll
          for (int i_76 = 0; i_76 < 8; ++i_76) {
            int condval_35;
            if ((0 <= ((((tile_4 * 64) + (i_76 * 8)) + (((int)threadIdx.x) >> 4)) + begin_4))) {
              condval_35 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_4 * 64) + (i_76 * 8)) + (((int)threadIdx.x) >> 4)) + begin_4), 173665))];
            } else {
              condval_35 = 0;
            }
            int token_4 = condval_35;
            bfloat16_t broadcast_var_33 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_36;
            if ((((((((tile_4 * 64) + (i_76 * 8)) + (((int)threadIdx.x) >> 4)) + begin_4) < end_4) && (0 <= token_4)) && (token_4 < 173666))) {
              condval_36 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_4) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_36 = make_uint4(__pack_nv_bfloat162(broadcast_var_33, broadcast_var_33), __pack_nv_bfloat162(broadcast_var_33, broadcast_var_33), __pack_nv_bfloat162(broadcast_var_33, broadcast_var_33), __pack_nv_bfloat162(broadcast_var_33, broadcast_var_33));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_76 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_36;
            bfloat16_t broadcast_var_34 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_37;
            if ((((((((tile_4 * 64) + (i_76 * 8)) + (((int)threadIdx.x) >> 4)) + begin_4) < end_4) && (0 <= token_4)) && (token_4 < 173666))) {
              condval_37 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_4) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_37 = make_uint4(__pack_nv_bfloat162(broadcast_var_34, broadcast_var_34), __pack_nv_bfloat162(broadcast_var_34, broadcast_var_34), __pack_nv_bfloat162(broadcast_var_34, broadcast_var_34), __pack_nv_bfloat162(broadcast_var_34, broadcast_var_34));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_76 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_37;
          }
          #pragma unroll
          for (int i_77 = 0; i_77 < 2; ++i_77) {
            float broadcast_var_35 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_4 + (i_77 * 4)) = make_float4(broadcast_var_35, broadcast_var_35, broadcast_var_35, broadcast_var_35);
          }
          {
            bfloat16_t A_local_8[8];
            bfloat16_t B_local_8[8];
            __syncthreads();
            for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_8 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_8[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_8 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_8[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_4 + 0), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_4 + 4), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + 4));
            }
          }
          #pragma unroll
          for (int i_78 = 0; i_78 < 8; ++i_78) {
            float condval_38;
            if ((((((((tile_4 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_78 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_4) + (i_78 & 1)) < end_4)) {
              condval_38 = (scores_4[i_78] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_38 = -CUDART_INF_F;
            }
            scores_4[i_78] = condval_38;
          }
          *(float2*)(previous_4 + 0) = *(float2*)(maximum_4 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_79 = 0; i_79 < 2; ++i_79) {
            maximum_clear_4[i_79] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_8 = 0; rv_8 < 4; ++rv_8) {
              maximum_clear_4[i_79] = max(maximum_clear_4[i_79], scores_4[((((rv_8 & 1) * 4) + (i_79 * 2)) + (rv_8 >> 1))]);
            }
            maximum_clear_4[i_79] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_4[i_79], (&(((float*)buf_dyn_shmem)[14272])));
            maximum_clear_4[i_79] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_4[i_79]);
            maximum_4[i_79] = max(maximum_4[i_79], maximum_clear_4[i_79]);
          }
          #pragma unroll
          for (int i_80 = 0; i_80 < 2; ++i_80) {
            float condval_39;
            if ((previous_4[i_80] == -CUDART_INF_F)) {
              condval_39 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_39 = exp2f(((previous_4[i_80] - maximum_4[i_80]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_4[i_80] = condval_39;
          }
          #pragma unroll
          for (int i_81 = 0; i_81 < 8; ++i_81) {
            float condval_40;
            if ((scores_4[i_81] == -CUDART_INF_F)) {
              condval_40 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_40 = exp2f(((scores_4[i_81] - maximum_4[((i_81 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_4[i_81] = condval_40;
          }
          __syncthreads();
          #pragma unroll
          for (int i_82 = 0; i_82 < 2; ++i_82) {
            subtotal_4[i_82] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_9 = 0; rv_9 < 4; ++rv_9) {
              subtotal_4[i_82] = (subtotal_4[i_82] + scores_4[((((rv_9 & 1) * 4) + (i_82 * 2)) + (rv_9 >> 1))]);
            }
            subtotal_4[i_82] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_4[i_82], (&(((float*)buf_dyn_shmem)[12608])));
            subtotal_4[i_82] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_4[i_82]);
          }
          #pragma unroll
          for (int i_83 = 0; i_83 < 2; ++i_83) {
            denominator_4[i_83] = ((denominator_4[i_83] * factor_4[i_83]) + subtotal_4[i_83]);
          }
          #pragma unroll
          for (int i_84 = 0; i_84 < 16; ++i_84) {
            out_4[i_84] = (out_4[i_84] * factor_4[((i_84 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_85 = 0; i_85 < 4; ++i_85) {
            uint1 __5;
            float2 v__4 = *(float2*)(scores_4 + (i_85 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
            *(uint1*)(probs_local_cast_4 + 0) = __5;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_85 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_85 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_85 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_4 + 0);
          }
          {
            bfloat16_t A_local_9[8];
            bfloat16_t B_local_9[16];
            __syncthreads();
            for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_9[0])));
              for (int i_86 = 0; i_86 < 2; ++i_86) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_9 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_86) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_9[(i_86 * 8)])));
              }
              for (int j_4 = 0; j_4 < 2; ++j_4) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_4 + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + (j_4 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_4 + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + ((j_4 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_87 = 0; i_87 < 32; ++i_87) {
          if (i_87 < 8) {
            cu[i_87] = out_4[(((i_87 >> 1) * 4) + (i_87 & 1))];
          }
        }
        #pragma unroll
        for (int i_88 = 0; i_88 < 8; ++i_88) {
          cu[(i_88 + 16)] = out_4[((((i_88 >> 1) * 4) + (i_88 & 1)) + 2)];
        }
      }
      __syncthreads();
      if (begin_4 < end_4) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_89 = 0; i_89 < 2; ++i_89) {
            if (i_89 < 1) {
              ((float*)buf_dyn_shmem)[((i_89 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_4[i_89];
              ((float*)buf_dyn_shmem)[(((i_89 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_4[i_89];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_4[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_4[1];
        }
      }
    }
    __syncthreads();
    if (label == 6) {
      int begin_5 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_5 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_5 < end_5) {
        #pragma unroll
        for (int i_90 = 0; i_90 < 2; ++i_90) {
          bfloat16_t broadcast_var_36 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_42;
          if ((i_90 < 1)) {
            condval_42 = ((((int)threadIdx.x) >> 4) + 8);
          } else {
            condval_42 = (((i_90 * 8) + (((int)threadIdx.x) >> 4)) + 8);
          }
          uint4 condval_41;
          if ((0 <= condval_42)) {
            int64_t condval_43;
            if ((((int64_t)i_90) < (int64_t)1)) {
              condval_43 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)8);
            } else {
              condval_43 = (((((int64_t)i_90) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
            }
            int64_t condval_44;
            if ((((int64_t)i_90) < (int64_t)1)) {
              condval_44 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)8);
            } else {
              condval_44 = (((((int64_t)i_90) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
            }
            condval_41 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_44, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_41 = make_uint4(__pack_nv_bfloat162(broadcast_var_36, broadcast_var_36), __pack_nv_bfloat162(broadcast_var_36, broadcast_var_36), __pack_nv_bfloat162(broadcast_var_36, broadcast_var_36), __pack_nv_bfloat162(broadcast_var_36, broadcast_var_36));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_90 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_41;
        }
        #pragma unroll
        for (int i_91 = 0; i_91 < 4; ++i_91) {
          float broadcast_var_37 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_5 + (i_91 * 4)) = make_float4(broadcast_var_37, broadcast_var_37, broadcast_var_37, broadcast_var_37);
        }
        #pragma unroll
        for (int i_92 = 0; i_92 < 16; ++i_92) {
          if ((i_92 & 3) < 2) {
            out_5[i_92] = cu[((((i_92 >> 2) * 2) + (i_92 & 1)) + 8)];
          }
        }
        #pragma unroll
        for (int i_93 = 0; i_93 < 8; ++i_93) {
          out_5[((((i_93 >> 1) * 4) + (i_93 & 1)) + 2)] = cu[(i_93 + 16)];
        }
        float broadcast_var_38 = -CUDART_INF_F;
        *(float2*)(maximum_5 + 0) = make_float2(broadcast_var_38, broadcast_var_38);
        float broadcast_var_39 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_5 + 0) = make_float2(broadcast_var_39, broadcast_var_39);
        #pragma unroll
        for (int i_94 = 0; i_94 < 2; ++i_94) {
          if (i_94 < 1) {
            maximum_5[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
            denominator_5[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
          }
        }
        maximum_5[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
        denominator_5[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
        for (int tile_5 = 0; tile_5 < (((end_5 + 63) - begin_5) >> 6); ++tile_5) {
          __syncthreads();
          #pragma unroll
          for (int i_95 = 0; i_95 < 8; ++i_95) {
            int condval_45;
            if ((0 <= ((((tile_5 * 64) + (i_95 * 8)) + (((int)threadIdx.x) >> 4)) + begin_5))) {
              condval_45 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_5 * 64) + (i_95 * 8)) + (((int)threadIdx.x) >> 4)) + begin_5), 173665))];
            } else {
              condval_45 = 0;
            }
            int token_5 = condval_45;
            bfloat16_t broadcast_var_40 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_46;
            if ((((((((tile_5 * 64) + (i_95 * 8)) + (((int)threadIdx.x) >> 4)) + begin_5) < end_5) && (0 <= token_5)) && (token_5 < 173666))) {
              condval_46 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_5) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_46 = make_uint4(__pack_nv_bfloat162(broadcast_var_40, broadcast_var_40), __pack_nv_bfloat162(broadcast_var_40, broadcast_var_40), __pack_nv_bfloat162(broadcast_var_40, broadcast_var_40), __pack_nv_bfloat162(broadcast_var_40, broadcast_var_40));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_95 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_46;
            bfloat16_t broadcast_var_41 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_47;
            if ((((((((tile_5 * 64) + (i_95 * 8)) + (((int)threadIdx.x) >> 4)) + begin_5) < end_5) && (0 <= token_5)) && (token_5 < 173666))) {
              condval_47 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_5) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_47 = make_uint4(__pack_nv_bfloat162(broadcast_var_41, broadcast_var_41), __pack_nv_bfloat162(broadcast_var_41, broadcast_var_41), __pack_nv_bfloat162(broadcast_var_41, broadcast_var_41), __pack_nv_bfloat162(broadcast_var_41, broadcast_var_41));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_95 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_47;
          }
          #pragma unroll
          for (int i_96 = 0; i_96 < 2; ++i_96) {
            float broadcast_var_42 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_5 + (i_96 * 4)) = make_float4(broadcast_var_42, broadcast_var_42, broadcast_var_42, broadcast_var_42);
          }
          {
            bfloat16_t A_local_10[8];
            bfloat16_t B_local_10[8];
            __syncthreads();
            for (int ki_10 = 0; ki_10 < 8; ++ki_10) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_10 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_10[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_10 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_10[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_5 + 0), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_5 + 4), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + 4));
            }
          }
          #pragma unroll
          for (int i_97 = 0; i_97 < 8; ++i_97) {
            float condval_48;
            if ((((((((tile_5 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_97 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_5) + (i_97 & 1)) < end_5)) {
              condval_48 = (scores_5[i_97] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_48 = -CUDART_INF_F;
            }
            scores_5[i_97] = condval_48;
          }
          *(float2*)(previous_5 + 0) = *(float2*)(maximum_5 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_98 = 0; i_98 < 2; ++i_98) {
            maximum_clear_5[i_98] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_10 = 0; rv_10 < 4; ++rv_10) {
              maximum_clear_5[i_98] = max(maximum_clear_5[i_98], scores_5[((((rv_10 & 1) * 4) + (i_98 * 2)) + (rv_10 >> 1))]);
            }
            maximum_clear_5[i_98] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_5[i_98], (&(((float*)buf_dyn_shmem)[12480])));
            maximum_clear_5[i_98] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_5[i_98]);
            maximum_5[i_98] = max(maximum_5[i_98], maximum_clear_5[i_98]);
          }
          #pragma unroll
          for (int i_99 = 0; i_99 < 2; ++i_99) {
            float condval_49;
            if ((previous_5[i_99] == -CUDART_INF_F)) {
              condval_49 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_49 = exp2f(((previous_5[i_99] - maximum_5[i_99]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_5[i_99] = condval_49;
          }
          #pragma unroll
          for (int i_100 = 0; i_100 < 8; ++i_100) {
            float condval_50;
            if ((scores_5[i_100] == -CUDART_INF_F)) {
              condval_50 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_50 = exp2f(((scores_5[i_100] - maximum_5[((i_100 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_5[i_100] = condval_50;
          }
          __syncthreads();
          #pragma unroll
          for (int i_101 = 0; i_101 < 2; ++i_101) {
            subtotal_5[i_101] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_11 = 0; rv_11 < 4; ++rv_11) {
              subtotal_5[i_101] = (subtotal_5[i_101] + scores_5[((((rv_11 & 1) * 4) + (i_101 * 2)) + (rv_11 >> 1))]);
            }
            subtotal_5[i_101] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_5[i_101], (&(((float*)buf_dyn_shmem)[12224])));
            subtotal_5[i_101] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_5[i_101]);
          }
          #pragma unroll
          for (int i_102 = 0; i_102 < 2; ++i_102) {
            denominator_5[i_102] = ((denominator_5[i_102] * factor_5[i_102]) + subtotal_5[i_102]);
          }
          #pragma unroll
          for (int i_103 = 0; i_103 < 16; ++i_103) {
            out_5[i_103] = (out_5[i_103] * factor_5[((i_103 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_104 = 0; i_104 < 4; ++i_104) {
            uint1 __6;
            float2 v__5 = *(float2*)(scores_5 + (i_104 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
            *(uint1*)(probs_local_cast_5 + 0) = __6;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_104 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_104 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_104 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_5 + 0);
          }
          {
            bfloat16_t A_local_11[8];
            bfloat16_t B_local_11[16];
            __syncthreads();
            for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_11[0])));
              for (int i_105 = 0; i_105 < 2; ++i_105) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_11 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_105) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_11[(i_105 * 8)])));
              }
              for (int j_5 = 0; j_5 < 2; ++j_5) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_5 + (j_5 * 8)), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + (j_5 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_5 + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + ((j_5 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_106 = 0; i_106 < 8; ++i_106) {
          cu[(i_106 + 8)] = out_5[(((i_106 >> 1) * 4) + (i_106 & 1))];
        }
        #pragma unroll
        for (int i_107 = 0; i_107 < 8; ++i_107) {
          cu[(i_107 + 16)] = out_5[((((i_107 >> 1) * 4) + (i_107 & 1)) + 2)];
        }
      }
      __syncthreads();
      if (begin_5 < end_5) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_108 = 0; i_108 < 2; ++i_108) {
            if (i_108 < 1) {
              ((float*)buf_dyn_shmem)[(((i_108 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 8)] = maximum_5[i_108];
              ((float*)buf_dyn_shmem)[(((i_108 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 40)] = denominator_5[i_108];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_5[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_5[1];
        }
      }
    }
    __syncthreads();
    if (label == 7) {
      int begin_6 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_6 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_6 < end_6) {
        #pragma unroll
        for (int i_109 = 0; i_109 < 4; ++i_109) {
          bfloat16_t broadcast_var_43 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_52;
          if ((i_109 < 1)) {
            condval_52 = (((int)threadIdx.x) >> 4);
          } else {
            int condval_53;
            if ((i_109 < 2)) {
              condval_53 = ((i_109 * 8) + (((int)threadIdx.x) >> 4));
            } else {
              condval_53 = ((i_109 * 8) + (((int)threadIdx.x) >> 4));
            }
            condval_52 = condval_53;
          }
          uint4 condval_51;
          if (((i_109 < 3) && (0 <= condval_52))) {
            int64_t condval_54;
            if ((((int64_t)i_109) < (int64_t)1)) {
              condval_54 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              int64_t condval_55;
              if ((((int64_t)i_109) < (int64_t)2)) {
                condval_55 = ((((int64_t)i_109) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
              } else {
                condval_55 = ((((int64_t)i_109) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
              }
              condval_54 = condval_55;
            }
            int64_t condval_56;
            if ((((int64_t)i_109) < (int64_t)1)) {
              condval_56 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              int64_t condval_57;
              if ((((int64_t)i_109) < (int64_t)2)) {
                condval_57 = ((((int64_t)i_109) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
              } else {
                condval_57 = ((((int64_t)i_109) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
              }
              condval_56 = condval_57;
            }
            condval_51 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_56, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_51 = make_uint4(__pack_nv_bfloat162(broadcast_var_43, broadcast_var_43), __pack_nv_bfloat162(broadcast_var_43, broadcast_var_43), __pack_nv_bfloat162(broadcast_var_43, broadcast_var_43), __pack_nv_bfloat162(broadcast_var_43, broadcast_var_43));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_109 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_51;
        }
        #pragma unroll
        for (int i_110 = 0; i_110 < 8; ++i_110) {
          float broadcast_var_44 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_6 + (i_110 * 4)) = make_float4(broadcast_var_44, broadcast_var_44, broadcast_var_44, broadcast_var_44);
        }
        #pragma unroll
        for (int i_111 = 0; i_111 < 32; ++i_111) {
          if ((((i_111 >> 4) * 2) + ((i_111 & 3) >> 1)) < 1) {
            out_6[i_111] = cu[(((((i_111 >> 4) * 16) + (((i_111 & 3) >> 1) * 8)) + (((i_111 & 15) >> 2) * 2)) + (i_111 & 1))];
          }
        }
        #pragma unroll
        for (int i_112 = 0; i_112 < 8; ++i_112) {
          out_6[((((i_112 >> 1) * 4) + (i_112 & 1)) + 2)] = cu[(i_112 + 8)];
        }
        #pragma unroll
        for (int i_113 = 0; i_113 < 8; ++i_113) {
          out_6[((((i_113 >> 1) * 4) + (i_113 & 1)) + 16)] = cu[(i_113 + 16)];
        }
        float broadcast_var_45 = -CUDART_INF_F;
        *(float4*)(maximum_6 + 0) = make_float4(broadcast_var_45, broadcast_var_45, broadcast_var_45, broadcast_var_45);
        float broadcast_var_46 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(denominator_6 + 0) = make_float4(broadcast_var_46, broadcast_var_46, broadcast_var_46, broadcast_var_46);
        #pragma unroll
        for (int i_114 = 0; i_114 < 4; ++i_114) {
          if (i_114 < 1) {
            maximum_6[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
            denominator_6[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
          }
        }
        maximum_6[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
        denominator_6[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
        maximum_6[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
        denominator_6[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
        for (int tile_6 = 0; tile_6 < (((end_6 + 63) - begin_6) >> 6); ++tile_6) {
          __syncthreads();
          #pragma unroll
          for (int i_115 = 0; i_115 < 8; ++i_115) {
            int condval_58;
            if ((0 <= ((((tile_6 * 64) + (i_115 * 8)) + (((int)threadIdx.x) >> 4)) + begin_6))) {
              condval_58 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_6 * 64) + (i_115 * 8)) + (((int)threadIdx.x) >> 4)) + begin_6), 173665))];
            } else {
              condval_58 = 0;
            }
            int token_6 = condval_58;
            bfloat16_t broadcast_var_47 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_59;
            if ((((((((tile_6 * 64) + (i_115 * 8)) + (((int)threadIdx.x) >> 4)) + begin_6) < end_6) && (0 <= token_6)) && (token_6 < 173666))) {
              condval_59 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_6) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_59 = make_uint4(__pack_nv_bfloat162(broadcast_var_47, broadcast_var_47), __pack_nv_bfloat162(broadcast_var_47, broadcast_var_47), __pack_nv_bfloat162(broadcast_var_47, broadcast_var_47), __pack_nv_bfloat162(broadcast_var_47, broadcast_var_47));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_115 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_59;
            bfloat16_t broadcast_var_48 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_60;
            if ((((((((tile_6 * 64) + (i_115 * 8)) + (((int)threadIdx.x) >> 4)) + begin_6) < end_6) && (0 <= token_6)) && (token_6 < 173666))) {
              condval_60 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_6) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_60 = make_uint4(__pack_nv_bfloat162(broadcast_var_48, broadcast_var_48), __pack_nv_bfloat162(broadcast_var_48, broadcast_var_48), __pack_nv_bfloat162(broadcast_var_48, broadcast_var_48), __pack_nv_bfloat162(broadcast_var_48, broadcast_var_48));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_115 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_60;
          }
          #pragma unroll
          for (int i_116 = 0; i_116 < 4; ++i_116) {
            float broadcast_var_49 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_6 + (i_116 * 4)) = make_float4(broadcast_var_49, broadcast_var_49, broadcast_var_49, broadcast_var_49);
          }
          {
            bfloat16_t A_local_12[16];
            bfloat16_t B_local_12[8];
            __syncthreads();
            for (int ki_12 = 0; ki_12 < 8; ++ki_12) {
              for (int i_117 = 0; i_117 < 2; ++i_117) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_12 >> 2) * 2048) + (i_117 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_12 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_12[(i_117 * 8)])));
              }
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_12 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_12 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_12[0])));
              for (int i_118 = 0; i_118 < 2; ++i_118) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_6 + (i_118 * 8)), reinterpret_cast<const unsigned*>(A_local_12 + (i_118 * 8)), reinterpret_cast<const unsigned*>(B_local_12 + 0));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_6 + ((i_118 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_12 + (i_118 * 8)), reinterpret_cast<const unsigned*>(B_local_12 + 4));
              }
            }
          }
          #pragma unroll
          for (int i_119 = 0; i_119 < 16; ++i_119) {
            float condval_61;
            if ((((((i_119 >> 3) * 2) + ((i_119 & 3) >> 1)) < 3) && (((((((tile_6 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_119 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_6) + (i_119 & 1)) < end_6))) {
              condval_61 = (scores_6[i_119] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_61 = -CUDART_INF_F;
            }
            scores_6[i_119] = condval_61;
          }
          *(float4*)(previous_6 + 0) = *(float4*)(maximum_6 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_120 = 0; i_120 < 4; ++i_120) {
            maximum_clear_6[i_120] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_12 = 0; rv_12 < 4; ++rv_12) {
              maximum_clear_6[i_120] = max(maximum_clear_6[i_120], scores_6[(((((i_120 >> 1) * 8) + ((rv_12 & 1) * 4)) + ((i_120 & 1) * 2)) + (rv_12 >> 1))]);
            }
            maximum_clear_6[i_120] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_6[i_120], (&(((float*)buf_dyn_shmem)[13888])));
            maximum_clear_6[i_120] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_6[i_120]);
            maximum_6[i_120] = max(maximum_6[i_120], maximum_clear_6[i_120]);
          }
          #pragma unroll
          for (int i_121 = 0; i_121 < 4; ++i_121) {
            float condval_62;
            if ((previous_6[i_121] == -CUDART_INF_F)) {
              condval_62 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_62 = exp2f(((previous_6[i_121] - maximum_6[i_121]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_6[i_121] = condval_62;
          }
          #pragma unroll
          for (int i_122 = 0; i_122 < 16; ++i_122) {
            float condval_63;
            if ((scores_6[i_122] == -CUDART_INF_F)) {
              condval_63 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_63 = exp2f(((scores_6[i_122] - maximum_6[(((i_122 >> 3) * 2) + ((i_122 & 3) >> 1))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_6[i_122] = condval_63;
          }
          __syncthreads();
          #pragma unroll
          for (int i_123 = 0; i_123 < 4; ++i_123) {
            subtotal_6[i_123] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_13 = 0; rv_13 < 4; ++rv_13) {
              subtotal_6[i_123] = (subtotal_6[i_123] + scores_6[(((((i_123 >> 1) * 8) + ((rv_13 & 1) * 4)) + ((i_123 & 1) * 2)) + (rv_13 >> 1))]);
            }
            subtotal_6[i_123] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_6[i_123], (&(((float*)buf_dyn_shmem)[13120])));
            subtotal_6[i_123] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_6[i_123]);
          }
          #pragma unroll
          for (int i_124 = 0; i_124 < 4; ++i_124) {
            denominator_6[i_124] = ((denominator_6[i_124] * factor_6[i_124]) + subtotal_6[i_124]);
          }
          #pragma unroll
          for (int i_125 = 0; i_125 < 32; ++i_125) {
            out_6[i_125] = (out_6[i_125] * factor_6[(((i_125 >> 4) * 2) + ((i_125 & 3) >> 1))]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_126 = 0; i_126 < 2; ++i_126) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_126 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), __pack_half2(((bfloat16_t)scores_6[(i_126 * 8)]), ((bfloat16_t)scores_6[((i_126 * 8) + 1)])), __pack_half2(((bfloat16_t)scores_6[((i_126 * 8) + 2)]), ((bfloat16_t)scores_6[((i_126 * 8) + 3)])), __pack_half2(((bfloat16_t)scores_6[((i_126 * 8) + 4)]), ((bfloat16_t)scores_6[((i_126 * 8) + 5)])), __pack_half2(((bfloat16_t)scores_6[((i_126 * 8) + 6)]), ((bfloat16_t)scores_6[((i_126 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_13[16];
            bfloat16_t B_local_13[16];
            __syncthreads();
            for (int ki_13 = 0; ki_13 < 4; ++ki_13) {
              for (int i_127 = 0; i_127 < 2; ++i_127) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_127 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_13[(i_127 * 8)])));
              }
              for (int i_128 = 0; i_128 < 2; ++i_128) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_13 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_128) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_13[(i_128 * 8)])));
              }
              for (int i_129 = 0; i_129 < 2; ++i_129) {
                for (int j_6 = 0; j_6 < 2; ++j_6) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_6 + ((i_129 * 16) + (j_6 * 8))), reinterpret_cast<const unsigned*>(A_local_13 + (i_129 * 8)), reinterpret_cast<const unsigned*>(B_local_13 + (j_6 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_6 + (((i_129 * 16) + (j_6 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_13 + (i_129 * 8)), reinterpret_cast<const unsigned*>(B_local_13 + ((j_6 * 8) + 4)));
                }
              }
            }
          }
        }
        #pragma unroll
        for (int i_130 = 0; i_130 < 32; ++i_130) {
          if (i_130 < 8) {
            cu[i_130] = out_6[(((i_130 >> 1) * 4) + (i_130 & 1))];
          }
        }
        #pragma unroll
        for (int i_131 = 0; i_131 < 8; ++i_131) {
          cu[(i_131 + 8)] = out_6[((((i_131 >> 1) * 4) + (i_131 & 1)) + 2)];
        }
        #pragma unroll
        for (int i_132 = 0; i_132 < 8; ++i_132) {
          cu[(i_132 + 16)] = out_6[((((i_132 >> 1) * 4) + (i_132 & 1)) + 16)];
        }
      }
      __syncthreads();
      if (begin_6 < end_6) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_133 = 0; i_133 < 4; ++i_133) {
            if (i_133 < 1) {
              ((float*)buf_dyn_shmem)[((i_133 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_6[i_133];
              ((float*)buf_dyn_shmem)[(((i_133 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_6[i_133];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)] = maximum_6[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)] = denominator_6[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_6[2];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_6[2];
        }
      }
    }
    if (label == 8) {
      int begin_7 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_7 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_7 < end_7) {
        #pragma unroll
        for (int i_134 = 0; i_134 < 2; ++i_134) {
          bfloat16_t broadcast_var_50 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_64;
          if ((i_134 < 1)) {
            condval_64 = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + (i_134 * 1024)) + (((int)threadIdx.x) * 8)) + 3072));
          } else {
            condval_64 = make_uint4(__pack_nv_bfloat162(broadcast_var_50, broadcast_var_50), __pack_nv_bfloat162(broadcast_var_50, broadcast_var_50), __pack_nv_bfloat162(broadcast_var_50, broadcast_var_50), __pack_nv_bfloat162(broadcast_var_50, broadcast_var_50));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_134 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_64;
        }
        #pragma unroll
        for (int i_135 = 0; i_135 < 4; ++i_135) {
          float broadcast_var_51 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_7 + (i_135 * 4)) = make_float4(broadcast_var_51, broadcast_var_51, broadcast_var_51, broadcast_var_51);
        }
        #pragma unroll
        for (int i_136 = 0; i_136 < 16; ++i_136) {
          if ((i_136 & 3) < 2) {
            out_7[i_136] = cu[((((i_136 >> 2) * 2) + (i_136 & 1)) + 24)];
          }
        }
        float broadcast_var_52 = -CUDART_INF_F;
        *(float2*)(maximum_7 + 0) = make_float2(broadcast_var_52, broadcast_var_52);
        float broadcast_var_53 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_7 + 0) = make_float2(broadcast_var_53, broadcast_var_53);
        #pragma unroll
        for (int i_137 = 0; i_137 < 2; ++i_137) {
          if (i_137 < 1) {
            maximum_7[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)];
            denominator_7[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)];
          }
        }
        for (int tile_7 = 0; tile_7 < (((end_7 + 63) - begin_7) >> 6); ++tile_7) {
          __syncthreads();
          #pragma unroll
          for (int i_138 = 0; i_138 < 8; ++i_138) {
            int condval_65;
            if ((0 <= ((((tile_7 * 64) + (i_138 * 8)) + (((int)threadIdx.x) >> 4)) + begin_7))) {
              condval_65 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_7 * 64) + (i_138 * 8)) + (((int)threadIdx.x) >> 4)) + begin_7), 173665))];
            } else {
              condval_65 = 0;
            }
            int token_7 = condval_65;
            bfloat16_t broadcast_var_54 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_66;
            if ((((((((tile_7 * 64) + (i_138 * 8)) + (((int)threadIdx.x) >> 4)) + begin_7) < end_7) && (0 <= token_7)) && (token_7 < 173666))) {
              condval_66 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_66 = make_uint4(__pack_nv_bfloat162(broadcast_var_54, broadcast_var_54), __pack_nv_bfloat162(broadcast_var_54, broadcast_var_54), __pack_nv_bfloat162(broadcast_var_54, broadcast_var_54), __pack_nv_bfloat162(broadcast_var_54, broadcast_var_54));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_138 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_66;
            bfloat16_t broadcast_var_55 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_67;
            if ((((((((tile_7 * 64) + (i_138 * 8)) + (((int)threadIdx.x) >> 4)) + begin_7) < end_7) && (0 <= token_7)) && (token_7 < 173666))) {
              condval_67 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_67 = make_uint4(__pack_nv_bfloat162(broadcast_var_55, broadcast_var_55), __pack_nv_bfloat162(broadcast_var_55, broadcast_var_55), __pack_nv_bfloat162(broadcast_var_55, broadcast_var_55), __pack_nv_bfloat162(broadcast_var_55, broadcast_var_55));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_138 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_67;
          }
          #pragma unroll
          for (int i_139 = 0; i_139 < 2; ++i_139) {
            float broadcast_var_56 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_7 + (i_139 * 4)) = make_float4(broadcast_var_56, broadcast_var_56, broadcast_var_56, broadcast_var_56);
          }
          {
            bfloat16_t A_local_14[8];
            bfloat16_t B_local_14[8];
            __syncthreads();
            for (int ki_14 = 0; ki_14 < 8; ++ki_14) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_14 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_14 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_14[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_14 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_14 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_14[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_7 + 0), reinterpret_cast<const unsigned*>(A_local_14 + 0), reinterpret_cast<const unsigned*>(B_local_14 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_7 + 4), reinterpret_cast<const unsigned*>(A_local_14 + 0), reinterpret_cast<const unsigned*>(B_local_14 + 4));
            }
          }
          #pragma unroll
          for (int i_140 = 0; i_140 < 8; ++i_140) {
            float condval_68;
            if ((((i_140 & 3) < 2) && (((((((tile_7 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_140 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_7) + (i_140 & 1)) < end_7))) {
              condval_68 = (scores_7[i_140] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_68 = -CUDART_INF_F;
            }
            scores_7[i_140] = condval_68;
          }
          *(float2*)(previous_7 + 0) = *(float2*)(maximum_7 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_141 = 0; i_141 < 2; ++i_141) {
            maximum_clear_7[i_141] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_14 = 0; rv_14 < 4; ++rv_14) {
              maximum_clear_7[i_141] = max(maximum_clear_7[i_141], scores_7[((((rv_14 & 1) * 4) + (i_141 * 2)) + (rv_14 >> 1))]);
            }
            maximum_clear_7[i_141] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_7[i_141], (&(((float*)buf_dyn_shmem)[12096])));
            maximum_clear_7[i_141] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_7[i_141]);
            maximum_7[i_141] = max(maximum_7[i_141], maximum_clear_7[i_141]);
          }
          #pragma unroll
          for (int i_142 = 0; i_142 < 2; ++i_142) {
            float condval_69;
            if ((previous_7[i_142] == -CUDART_INF_F)) {
              condval_69 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_69 = exp2f(((previous_7[i_142] - maximum_7[i_142]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_7[i_142] = condval_69;
          }
          #pragma unroll
          for (int i_143 = 0; i_143 < 8; ++i_143) {
            float condval_70;
            if ((scores_7[i_143] == -CUDART_INF_F)) {
              condval_70 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_70 = exp2f(((scores_7[i_143] - maximum_7[((i_143 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_7[i_143] = condval_70;
          }
          __syncthreads();
          #pragma unroll
          for (int i_144 = 0; i_144 < 2; ++i_144) {
            subtotal_7[i_144] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_15 = 0; rv_15 < 4; ++rv_15) {
              subtotal_7[i_144] = (subtotal_7[i_144] + scores_7[((((rv_15 & 1) * 4) + (i_144 * 2)) + (rv_15 >> 1))]);
            }
            subtotal_7[i_144] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_7[i_144], (&(((float*)buf_dyn_shmem)[11968])));
            subtotal_7[i_144] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_7[i_144]);
          }
          #pragma unroll
          for (int i_145 = 0; i_145 < 2; ++i_145) {
            denominator_7[i_145] = ((denominator_7[i_145] * factor_7[i_145]) + subtotal_7[i_145]);
          }
          #pragma unroll
          for (int i_146 = 0; i_146 < 16; ++i_146) {
            out_7[i_146] = (out_7[i_146] * factor_7[((i_146 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_147 = 0; i_147 < 4; ++i_147) {
            uint1 __7;
            float2 v__6 = *(float2*)(scores_7 + (i_147 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&v__6))[0]);
            *(uint1*)(probs_local_cast_6 + 0) = __7;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_147 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_147 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_147 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_6 + 0);
          }
          {
            bfloat16_t A_local_15[8];
            bfloat16_t B_local_15[16];
            __syncthreads();
            for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_15 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_15 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_15[0])));
              for (int i_148 = 0; i_148 < 2; ++i_148) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_15 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_148) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_15[(i_148 * 8)])));
              }
              for (int j_7 = 0; j_7 < 2; ++j_7) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_7 + (j_7 * 8)), reinterpret_cast<const unsigned*>(A_local_15 + 0), reinterpret_cast<const unsigned*>(B_local_15 + (j_7 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_7 + ((j_7 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_15 + 0), reinterpret_cast<const unsigned*>(B_local_15 + ((j_7 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_149 = 0; i_149 < 8; ++i_149) {
          cu[(i_149 + 24)] = out_7[(((i_149 >> 1) * 4) + (i_149 & 1))];
        }
      }
      __syncthreads();
      if (begin_7 < end_7) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_150 = 0; i_150 < 2; ++i_150) {
            if (i_150 < 1) {
              ((float*)buf_dyn_shmem)[(((i_150 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 24)] = maximum_7[i_150];
              ((float*)buf_dyn_shmem)[(((i_150 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 56)] = denominator_7[i_150];
            }
          }
        }
      }
    }
    if (label == 9) {
      int begin_8 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_8 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_8 < end_8) {
        #pragma unroll
        for (int i_151 = 0; i_151 < 2; ++i_151) {
          bfloat16_t broadcast_var_57 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_72;
          if ((i_151 < 1)) {
            condval_72 = (((int)threadIdx.x) >> 4);
          } else {
            condval_72 = (((i_151 * 8) + (((int)threadIdx.x) >> 4)) + 16);
          }
          uint4 condval_71;
          if ((0 <= condval_72)) {
            int64_t condval_73;
            if ((((int64_t)i_151) < (int64_t)1)) {
              condval_73 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              condval_73 = (((((int64_t)i_151) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)16);
            }
            int64_t condval_74;
            if ((((int64_t)i_151) < (int64_t)1)) {
              condval_74 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              condval_74 = (((((int64_t)i_151) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)16);
            }
            condval_71 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_74, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_71 = make_uint4(__pack_nv_bfloat162(broadcast_var_57, broadcast_var_57), __pack_nv_bfloat162(broadcast_var_57, broadcast_var_57), __pack_nv_bfloat162(broadcast_var_57, broadcast_var_57), __pack_nv_bfloat162(broadcast_var_57, broadcast_var_57));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_151 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_71;
        }
        #pragma unroll
        for (int i_152 = 0; i_152 < 4; ++i_152) {
          float broadcast_var_58 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_8 + (i_152 * 4)) = make_float4(broadcast_var_58, broadcast_var_58, broadcast_var_58, broadcast_var_58);
        }
        #pragma unroll
        for (int i_153 = 0; i_153 < 16; ++i_153) {
          if ((i_153 & 3) < 2) {
            out_8[i_153] = cu[(((i_153 >> 2) * 2) + (i_153 & 1))];
          }
        }
        #pragma unroll
        for (int i_154 = 0; i_154 < 8; ++i_154) {
          out_8[((((i_154 >> 1) * 4) + (i_154 & 1)) + 2)] = cu[(i_154 + 24)];
        }
        float broadcast_var_59 = -CUDART_INF_F;
        *(float2*)(maximum_8 + 0) = make_float2(broadcast_var_59, broadcast_var_59);
        float broadcast_var_60 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_8 + 0) = make_float2(broadcast_var_60, broadcast_var_60);
        #pragma unroll
        for (int i_155 = 0; i_155 < 2; ++i_155) {
          if (i_155 < 1) {
            maximum_8[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
            denominator_8[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
          }
        }
        maximum_8[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)];
        denominator_8[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)];
        for (int tile_8 = 0; tile_8 < (((end_8 + 63) - begin_8) >> 6); ++tile_8) {
          __syncthreads();
          #pragma unroll
          for (int i_156 = 0; i_156 < 8; ++i_156) {
            int condval_75;
            if ((0 <= ((((tile_8 * 64) + (i_156 * 8)) + (((int)threadIdx.x) >> 4)) + begin_8))) {
              condval_75 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_8 * 64) + (i_156 * 8)) + (((int)threadIdx.x) >> 4)) + begin_8), 173665))];
            } else {
              condval_75 = 0;
            }
            int token_8 = condval_75;
            bfloat16_t broadcast_var_61 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_76;
            if ((((((((tile_8 * 64) + (i_156 * 8)) + (((int)threadIdx.x) >> 4)) + begin_8) < end_8) && (0 <= token_8)) && (token_8 < 173666))) {
              condval_76 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_8) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_76 = make_uint4(__pack_nv_bfloat162(broadcast_var_61, broadcast_var_61), __pack_nv_bfloat162(broadcast_var_61, broadcast_var_61), __pack_nv_bfloat162(broadcast_var_61, broadcast_var_61), __pack_nv_bfloat162(broadcast_var_61, broadcast_var_61));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_156 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_76;
            bfloat16_t broadcast_var_62 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_77;
            if ((((((((tile_8 * 64) + (i_156 * 8)) + (((int)threadIdx.x) >> 4)) + begin_8) < end_8) && (0 <= token_8)) && (token_8 < 173666))) {
              condval_77 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_8) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_77 = make_uint4(__pack_nv_bfloat162(broadcast_var_62, broadcast_var_62), __pack_nv_bfloat162(broadcast_var_62, broadcast_var_62), __pack_nv_bfloat162(broadcast_var_62, broadcast_var_62), __pack_nv_bfloat162(broadcast_var_62, broadcast_var_62));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_156 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_77;
          }
          #pragma unroll
          for (int i_157 = 0; i_157 < 2; ++i_157) {
            float broadcast_var_63 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_8 + (i_157 * 4)) = make_float4(broadcast_var_63, broadcast_var_63, broadcast_var_63, broadcast_var_63);
          }
          {
            bfloat16_t A_local_16[8];
            bfloat16_t B_local_16[8];
            __syncthreads();
            for (int ki_16 = 0; ki_16 < 8; ++ki_16) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_16 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_16[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_16 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_16[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_8 + 0), reinterpret_cast<const unsigned*>(A_local_16 + 0), reinterpret_cast<const unsigned*>(B_local_16 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_8 + 4), reinterpret_cast<const unsigned*>(A_local_16 + 0), reinterpret_cast<const unsigned*>(B_local_16 + 4));
            }
          }
          #pragma unroll
          for (int i_158 = 0; i_158 < 8; ++i_158) {
            float condval_78;
            if ((((((((tile_8 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_158 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_8) + (i_158 & 1)) < end_8)) {
              condval_78 = (scores_8[i_158] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_78 = -CUDART_INF_F;
            }
            scores_8[i_158] = condval_78;
          }
          *(float2*)(previous_8 + 0) = *(float2*)(maximum_8 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_159 = 0; i_159 < 2; ++i_159) {
            maximum_clear_8[i_159] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_16 = 0; rv_16 < 4; ++rv_16) {
              maximum_clear_8[i_159] = max(maximum_clear_8[i_159], scores_8[((((rv_16 & 1) * 4) + (i_159 * 2)) + (rv_16 >> 1))]);
            }
            maximum_clear_8[i_159] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_8[i_159], (&(((float*)buf_dyn_shmem)[14016])));
            maximum_clear_8[i_159] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_8[i_159]);
            maximum_8[i_159] = max(maximum_8[i_159], maximum_clear_8[i_159]);
          }
          #pragma unroll
          for (int i_160 = 0; i_160 < 2; ++i_160) {
            float condval_79;
            if ((previous_8[i_160] == -CUDART_INF_F)) {
              condval_79 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_79 = exp2f(((previous_8[i_160] - maximum_8[i_160]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_8[i_160] = condval_79;
          }
          #pragma unroll
          for (int i_161 = 0; i_161 < 8; ++i_161) {
            float condval_80;
            if ((scores_8[i_161] == -CUDART_INF_F)) {
              condval_80 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_80 = exp2f(((scores_8[i_161] - maximum_8[((i_161 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_8[i_161] = condval_80;
          }
          __syncthreads();
          #pragma unroll
          for (int i_162 = 0; i_162 < 2; ++i_162) {
            subtotal_8[i_162] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_17 = 0; rv_17 < 4; ++rv_17) {
              subtotal_8[i_162] = (subtotal_8[i_162] + scores_8[((((rv_17 & 1) * 4) + (i_162 * 2)) + (rv_17 >> 1))]);
            }
            subtotal_8[i_162] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_8[i_162], (&(((float*)buf_dyn_shmem)[11840])));
            subtotal_8[i_162] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_8[i_162]);
          }
          #pragma unroll
          for (int i_163 = 0; i_163 < 2; ++i_163) {
            denominator_8[i_163] = ((denominator_8[i_163] * factor_8[i_163]) + subtotal_8[i_163]);
          }
          #pragma unroll
          for (int i_164 = 0; i_164 < 16; ++i_164) {
            out_8[i_164] = (out_8[i_164] * factor_8[((i_164 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_165 = 0; i_165 < 4; ++i_165) {
            uint1 __8;
            float2 v__7 = *(float2*)(scores_8 + (i_165 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__8))[0] = __float22bfloat162_rn(((float2*)(&v__7))[0]);
            *(uint1*)(probs_local_cast_7 + 0) = __8;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_165 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_165 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_165 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_7 + 0);
          }
          {
            bfloat16_t A_local_17[8];
            bfloat16_t B_local_17[16];
            __syncthreads();
            for (int ki_17 = 0; ki_17 < 4; ++ki_17) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_17[0])));
              for (int i_166 = 0; i_166 < 2; ++i_166) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_17 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_166) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_17[(i_166 * 8)])));
              }
              for (int j_8 = 0; j_8 < 2; ++j_8) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_8 + (j_8 * 8)), reinterpret_cast<const unsigned*>(A_local_17 + 0), reinterpret_cast<const unsigned*>(B_local_17 + (j_8 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_8 + ((j_8 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_17 + 0), reinterpret_cast<const unsigned*>(B_local_17 + ((j_8 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_167 = 0; i_167 < 32; ++i_167) {
          if (i_167 < 8) {
            cu[i_167] = out_8[(((i_167 >> 1) * 4) + (i_167 & 1))];
          }
        }
        #pragma unroll
        for (int i_168 = 0; i_168 < 8; ++i_168) {
          cu[(i_168 + 24)] = out_8[((((i_168 >> 1) * 4) + (i_168 & 1)) + 2)];
        }
      }
      __syncthreads();
      if (begin_8 < end_8) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_169 = 0; i_169 < 2; ++i_169) {
            if (i_169 < 1) {
              ((float*)buf_dyn_shmem)[((i_169 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_8[i_169];
              ((float*)buf_dyn_shmem)[(((i_169 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_8[i_169];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)] = maximum_8[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)] = denominator_8[1];
        }
      }
    }
    __syncthreads();
    if (label == 10) {
      int begin_9 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_9 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_9 < end_9) {
        #pragma unroll
        for (int i_170 = 0; i_170 < 2; ++i_170) {
          bfloat16_t broadcast_var_64 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_82;
          if ((i_170 < 1)) {
            condval_82 = ((((int)threadIdx.x) >> 4) + 8);
          } else {
            condval_82 = (((i_170 * 8) + (((int)threadIdx.x) >> 4)) + 16);
          }
          uint4 condval_81;
          if ((0 <= condval_82)) {
            int64_t condval_83;
            if ((((int64_t)i_170) < (int64_t)1)) {
              condval_83 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)8);
            } else {
              condval_83 = (((((int64_t)i_170) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)16);
            }
            int64_t condval_84;
            if ((((int64_t)i_170) < (int64_t)1)) {
              condval_84 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)8);
            } else {
              condval_84 = (((((int64_t)i_170) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)16);
            }
            condval_81 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_84, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_81 = make_uint4(__pack_nv_bfloat162(broadcast_var_64, broadcast_var_64), __pack_nv_bfloat162(broadcast_var_64, broadcast_var_64), __pack_nv_bfloat162(broadcast_var_64, broadcast_var_64), __pack_nv_bfloat162(broadcast_var_64, broadcast_var_64));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_170 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_81;
        }
        #pragma unroll
        for (int i_171 = 0; i_171 < 4; ++i_171) {
          float broadcast_var_65 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_9 + (i_171 * 4)) = make_float4(broadcast_var_65, broadcast_var_65, broadcast_var_65, broadcast_var_65);
        }
        #pragma unroll
        for (int i_172 = 0; i_172 < 16; ++i_172) {
          if ((i_172 & 3) < 2) {
            out_9[i_172] = cu[((((i_172 >> 2) * 2) + (i_172 & 1)) + 8)];
          }
        }
        #pragma unroll
        for (int i_173 = 0; i_173 < 8; ++i_173) {
          out_9[((((i_173 >> 1) * 4) + (i_173 & 1)) + 2)] = cu[(i_173 + 24)];
        }
        float broadcast_var_66 = -CUDART_INF_F;
        *(float2*)(maximum_9 + 0) = make_float2(broadcast_var_66, broadcast_var_66);
        float broadcast_var_67 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_9 + 0) = make_float2(broadcast_var_67, broadcast_var_67);
        #pragma unroll
        for (int i_174 = 0; i_174 < 2; ++i_174) {
          if (i_174 < 1) {
            maximum_9[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
            denominator_9[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
          }
        }
        maximum_9[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)];
        denominator_9[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)];
        for (int tile_9 = 0; tile_9 < (((end_9 + 63) - begin_9) >> 6); ++tile_9) {
          __syncthreads();
          #pragma unroll
          for (int i_175 = 0; i_175 < 8; ++i_175) {
            int condval_85;
            if ((0 <= ((((tile_9 * 64) + (i_175 * 8)) + (((int)threadIdx.x) >> 4)) + begin_9))) {
              condval_85 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_9 * 64) + (i_175 * 8)) + (((int)threadIdx.x) >> 4)) + begin_9), 173665))];
            } else {
              condval_85 = 0;
            }
            int token_9 = condval_85;
            bfloat16_t broadcast_var_68 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_86;
            if ((((((((tile_9 * 64) + (i_175 * 8)) + (((int)threadIdx.x) >> 4)) + begin_9) < end_9) && (0 <= token_9)) && (token_9 < 173666))) {
              condval_86 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_9) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_86 = make_uint4(__pack_nv_bfloat162(broadcast_var_68, broadcast_var_68), __pack_nv_bfloat162(broadcast_var_68, broadcast_var_68), __pack_nv_bfloat162(broadcast_var_68, broadcast_var_68), __pack_nv_bfloat162(broadcast_var_68, broadcast_var_68));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_175 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_86;
            bfloat16_t broadcast_var_69 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_87;
            if ((((((((tile_9 * 64) + (i_175 * 8)) + (((int)threadIdx.x) >> 4)) + begin_9) < end_9) && (0 <= token_9)) && (token_9 < 173666))) {
              condval_87 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_9) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_87 = make_uint4(__pack_nv_bfloat162(broadcast_var_69, broadcast_var_69), __pack_nv_bfloat162(broadcast_var_69, broadcast_var_69), __pack_nv_bfloat162(broadcast_var_69, broadcast_var_69), __pack_nv_bfloat162(broadcast_var_69, broadcast_var_69));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_175 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_87;
          }
          #pragma unroll
          for (int i_176 = 0; i_176 < 2; ++i_176) {
            float broadcast_var_70 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_9 + (i_176 * 4)) = make_float4(broadcast_var_70, broadcast_var_70, broadcast_var_70, broadcast_var_70);
          }
          {
            bfloat16_t A_local_18[8];
            bfloat16_t B_local_18[8];
            __syncthreads();
            for (int ki_18 = 0; ki_18 < 8; ++ki_18) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_18 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_18 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_18[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_18 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_18 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_18[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_9 + 0), reinterpret_cast<const unsigned*>(A_local_18 + 0), reinterpret_cast<const unsigned*>(B_local_18 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_9 + 4), reinterpret_cast<const unsigned*>(A_local_18 + 0), reinterpret_cast<const unsigned*>(B_local_18 + 4));
            }
          }
          #pragma unroll
          for (int i_177 = 0; i_177 < 8; ++i_177) {
            float condval_88;
            if ((((((((tile_9 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_177 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_9) + (i_177 & 1)) < end_9)) {
              condval_88 = (scores_9[i_177] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_88 = -CUDART_INF_F;
            }
            scores_9[i_177] = condval_88;
          }
          *(float2*)(previous_9 + 0) = *(float2*)(maximum_9 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_178 = 0; i_178 < 2; ++i_178) {
            maximum_clear_9[i_178] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_18 = 0; rv_18 < 4; ++rv_18) {
              maximum_clear_9[i_178] = max(maximum_clear_9[i_178], scores_9[((((rv_18 & 1) * 4) + (i_178 * 2)) + (rv_18 >> 1))]);
            }
            maximum_clear_9[i_178] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_9[i_178], (&(((float*)buf_dyn_shmem)[11712])));
            maximum_clear_9[i_178] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_9[i_178]);
            maximum_9[i_178] = max(maximum_9[i_178], maximum_clear_9[i_178]);
          }
          #pragma unroll
          for (int i_179 = 0; i_179 < 2; ++i_179) {
            float condval_89;
            if ((previous_9[i_179] == -CUDART_INF_F)) {
              condval_89 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_89 = exp2f(((previous_9[i_179] - maximum_9[i_179]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_9[i_179] = condval_89;
          }
          #pragma unroll
          for (int i_180 = 0; i_180 < 8; ++i_180) {
            float condval_90;
            if ((scores_9[i_180] == -CUDART_INF_F)) {
              condval_90 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_90 = exp2f(((scores_9[i_180] - maximum_9[((i_180 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_9[i_180] = condval_90;
          }
          __syncthreads();
          #pragma unroll
          for (int i_181 = 0; i_181 < 2; ++i_181) {
            subtotal_9[i_181] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_19 = 0; rv_19 < 4; ++rv_19) {
              subtotal_9[i_181] = (subtotal_9[i_181] + scores_9[((((rv_19 & 1) * 4) + (i_181 * 2)) + (rv_19 >> 1))]);
            }
            subtotal_9[i_181] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_9[i_181], (&(((float*)buf_dyn_shmem)[13504])));
            subtotal_9[i_181] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_9[i_181]);
          }
          #pragma unroll
          for (int i_182 = 0; i_182 < 2; ++i_182) {
            denominator_9[i_182] = ((denominator_9[i_182] * factor_9[i_182]) + subtotal_9[i_182]);
          }
          #pragma unroll
          for (int i_183 = 0; i_183 < 16; ++i_183) {
            out_9[i_183] = (out_9[i_183] * factor_9[((i_183 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_184 = 0; i_184 < 4; ++i_184) {
            uint1 __9;
            float2 v__8 = *(float2*)(scores_9 + (i_184 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__8))[0]);
            *(uint1*)(probs_local_cast_8 + 0) = __9;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_184 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_184 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_184 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_8 + 0);
          }
          {
            bfloat16_t A_local_19[8];
            bfloat16_t B_local_19[16];
            __syncthreads();
            for (int ki_19 = 0; ki_19 < 4; ++ki_19) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_19 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_19[0])));
              for (int i_185 = 0; i_185 < 2; ++i_185) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_19 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_185) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_19[(i_185 * 8)])));
              }
              for (int j_9 = 0; j_9 < 2; ++j_9) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_9 + (j_9 * 8)), reinterpret_cast<const unsigned*>(A_local_19 + 0), reinterpret_cast<const unsigned*>(B_local_19 + (j_9 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_9 + ((j_9 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_19 + 0), reinterpret_cast<const unsigned*>(B_local_19 + ((j_9 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_186 = 0; i_186 < 8; ++i_186) {
          cu[(i_186 + 8)] = out_9[(((i_186 >> 1) * 4) + (i_186 & 1))];
        }
        #pragma unroll
        for (int i_187 = 0; i_187 < 8; ++i_187) {
          cu[(i_187 + 24)] = out_9[((((i_187 >> 1) * 4) + (i_187 & 1)) + 2)];
        }
      }
      __syncthreads();
      if (begin_9 < end_9) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_188 = 0; i_188 < 2; ++i_188) {
            if (i_188 < 1) {
              ((float*)buf_dyn_shmem)[(((i_188 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 8)] = maximum_9[i_188];
              ((float*)buf_dyn_shmem)[(((i_188 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 40)] = denominator_9[i_188];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)] = maximum_9[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)] = denominator_9[1];
        }
      }
    }
    __syncthreads();
    if (label == 11) {
      int begin_10 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_10 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_10 < end_10) {
        #pragma unroll
        for (int i_189 = 0; i_189 < 4; ++i_189) {
          bfloat16_t broadcast_var_71 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_92;
          if ((i_189 < 1)) {
            condval_92 = (((int)threadIdx.x) >> 4);
          } else {
            int condval_93;
            if ((i_189 < 2)) {
              condval_93 = ((i_189 * 8) + (((int)threadIdx.x) >> 4));
            } else {
              condval_93 = (((i_189 * 8) + (((int)threadIdx.x) >> 4)) + 8);
            }
            condval_92 = condval_93;
          }
          uint4 condval_91;
          if (((i_189 < 3) && (0 <= condval_92))) {
            int64_t condval_94;
            if ((((int64_t)i_189) < (int64_t)1)) {
              condval_94 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              int64_t condval_95;
              if ((((int64_t)i_189) < (int64_t)2)) {
                condval_95 = ((((int64_t)i_189) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
              } else {
                condval_95 = (((((int64_t)i_189) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              }
              condval_94 = condval_95;
            }
            int64_t condval_96;
            if ((((int64_t)i_189) < (int64_t)1)) {
              condval_96 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              int64_t condval_97;
              if ((((int64_t)i_189) < (int64_t)2)) {
                condval_97 = ((((int64_t)i_189) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
              } else {
                condval_97 = (((((int64_t)i_189) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              }
              condval_96 = condval_97;
            }
            condval_91 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_96, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_91 = make_uint4(__pack_nv_bfloat162(broadcast_var_71, broadcast_var_71), __pack_nv_bfloat162(broadcast_var_71, broadcast_var_71), __pack_nv_bfloat162(broadcast_var_71, broadcast_var_71), __pack_nv_bfloat162(broadcast_var_71, broadcast_var_71));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_189 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_91;
        }
        #pragma unroll
        for (int i_190 = 0; i_190 < 8; ++i_190) {
          float broadcast_var_72 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_10 + (i_190 * 4)) = make_float4(broadcast_var_72, broadcast_var_72, broadcast_var_72, broadcast_var_72);
        }
        #pragma unroll
        for (int i_191 = 0; i_191 < 32; ++i_191) {
          if ((((i_191 >> 4) * 2) + ((i_191 & 3) >> 1)) < 1) {
            out_10[i_191] = cu[(((((i_191 >> 4) * 16) + (((i_191 & 3) >> 1) * 8)) + (((i_191 & 15) >> 2) * 2)) + (i_191 & 1))];
          }
        }
        #pragma unroll
        for (int i_192 = 0; i_192 < 8; ++i_192) {
          out_10[((((i_192 >> 1) * 4) + (i_192 & 1)) + 2)] = cu[(i_192 + 8)];
        }
        #pragma unroll
        for (int i_193 = 0; i_193 < 8; ++i_193) {
          out_10[((((i_193 >> 1) * 4) + (i_193 & 1)) + 16)] = cu[(i_193 + 24)];
        }
        float broadcast_var_73 = -CUDART_INF_F;
        *(float4*)(maximum_10 + 0) = make_float4(broadcast_var_73, broadcast_var_73, broadcast_var_73, broadcast_var_73);
        float broadcast_var_74 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(denominator_10 + 0) = make_float4(broadcast_var_74, broadcast_var_74, broadcast_var_74, broadcast_var_74);
        #pragma unroll
        for (int i_194 = 0; i_194 < 4; ++i_194) {
          if (i_194 < 1) {
            maximum_10[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
            denominator_10[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
          }
        }
        maximum_10[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
        denominator_10[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
        maximum_10[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)];
        denominator_10[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)];
        for (int tile_10 = 0; tile_10 < (((end_10 + 63) - begin_10) >> 6); ++tile_10) {
          __syncthreads();
          #pragma unroll
          for (int i_195 = 0; i_195 < 8; ++i_195) {
            int condval_98;
            if ((0 <= ((((tile_10 * 64) + (i_195 * 8)) + (((int)threadIdx.x) >> 4)) + begin_10))) {
              condval_98 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_10 * 64) + (i_195 * 8)) + (((int)threadIdx.x) >> 4)) + begin_10), 173665))];
            } else {
              condval_98 = 0;
            }
            int token_10 = condval_98;
            bfloat16_t broadcast_var_75 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_99;
            if ((((((((tile_10 * 64) + (i_195 * 8)) + (((int)threadIdx.x) >> 4)) + begin_10) < end_10) && (0 <= token_10)) && (token_10 < 173666))) {
              condval_99 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_10) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_99 = make_uint4(__pack_nv_bfloat162(broadcast_var_75, broadcast_var_75), __pack_nv_bfloat162(broadcast_var_75, broadcast_var_75), __pack_nv_bfloat162(broadcast_var_75, broadcast_var_75), __pack_nv_bfloat162(broadcast_var_75, broadcast_var_75));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_195 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_99;
            bfloat16_t broadcast_var_76 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_100;
            if ((((((((tile_10 * 64) + (i_195 * 8)) + (((int)threadIdx.x) >> 4)) + begin_10) < end_10) && (0 <= token_10)) && (token_10 < 173666))) {
              condval_100 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_10) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_100 = make_uint4(__pack_nv_bfloat162(broadcast_var_76, broadcast_var_76), __pack_nv_bfloat162(broadcast_var_76, broadcast_var_76), __pack_nv_bfloat162(broadcast_var_76, broadcast_var_76), __pack_nv_bfloat162(broadcast_var_76, broadcast_var_76));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_195 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_100;
          }
          #pragma unroll
          for (int i_196 = 0; i_196 < 4; ++i_196) {
            float broadcast_var_77 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_10 + (i_196 * 4)) = make_float4(broadcast_var_77, broadcast_var_77, broadcast_var_77, broadcast_var_77);
          }
          {
            bfloat16_t A_local_20[16];
            bfloat16_t B_local_20[8];
            __syncthreads();
            for (int ki_20 = 0; ki_20 < 8; ++ki_20) {
              for (int i_197 = 0; i_197 < 2; ++i_197) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_20 >> 2) * 2048) + (i_197 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_20 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_20[(i_197 * 8)])));
              }
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_20 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_20 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_20[0])));
              for (int i_198 = 0; i_198 < 2; ++i_198) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_10 + (i_198 * 8)), reinterpret_cast<const unsigned*>(A_local_20 + (i_198 * 8)), reinterpret_cast<const unsigned*>(B_local_20 + 0));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_10 + ((i_198 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_20 + (i_198 * 8)), reinterpret_cast<const unsigned*>(B_local_20 + 4));
              }
            }
          }
          #pragma unroll
          for (int i_199 = 0; i_199 < 16; ++i_199) {
            float condval_101;
            if ((((((i_199 >> 3) * 2) + ((i_199 & 3) >> 1)) < 3) && (((((((tile_10 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_199 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_10) + (i_199 & 1)) < end_10))) {
              condval_101 = (scores_10[i_199] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_101 = -CUDART_INF_F;
            }
            scores_10[i_199] = condval_101;
          }
          *(float4*)(previous_10 + 0) = *(float4*)(maximum_10 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_200 = 0; i_200 < 4; ++i_200) {
            maximum_clear_10[i_200] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_20 = 0; rv_20 < 4; ++rv_20) {
              maximum_clear_10[i_200] = max(maximum_clear_10[i_200], scores_10[(((((i_200 >> 1) * 8) + ((rv_20 & 1) * 4)) + ((i_200 & 1) * 2)) + (rv_20 >> 1))]);
            }
            maximum_clear_10[i_200] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_10[i_200], (&(((float*)buf_dyn_shmem)[12864])));
            maximum_clear_10[i_200] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_10[i_200]);
            maximum_10[i_200] = max(maximum_10[i_200], maximum_clear_10[i_200]);
          }
          #pragma unroll
          for (int i_201 = 0; i_201 < 4; ++i_201) {
            float condval_102;
            if ((previous_10[i_201] == -CUDART_INF_F)) {
              condval_102 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_102 = exp2f(((previous_10[i_201] - maximum_10[i_201]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_10[i_201] = condval_102;
          }
          #pragma unroll
          for (int i_202 = 0; i_202 < 16; ++i_202) {
            float condval_103;
            if ((scores_10[i_202] == -CUDART_INF_F)) {
              condval_103 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_103 = exp2f(((scores_10[i_202] - maximum_10[(((i_202 >> 3) * 2) + ((i_202 & 3) >> 1))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_10[i_202] = condval_103;
          }
          __syncthreads();
          #pragma unroll
          for (int i_203 = 0; i_203 < 4; ++i_203) {
            subtotal_10[i_203] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_21 = 0; rv_21 < 4; ++rv_21) {
              subtotal_10[i_203] = (subtotal_10[i_203] + scores_10[(((((i_203 >> 1) * 8) + ((rv_21 & 1) * 4)) + ((i_203 & 1) * 2)) + (rv_21 >> 1))]);
            }
            subtotal_10[i_203] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_10[i_203], (&(((float*)buf_dyn_shmem)[12736])));
            subtotal_10[i_203] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_10[i_203]);
          }
          #pragma unroll
          for (int i_204 = 0; i_204 < 4; ++i_204) {
            denominator_10[i_204] = ((denominator_10[i_204] * factor_10[i_204]) + subtotal_10[i_204]);
          }
          #pragma unroll
          for (int i_205 = 0; i_205 < 32; ++i_205) {
            out_10[i_205] = (out_10[i_205] * factor_10[(((i_205 >> 4) * 2) + ((i_205 & 3) >> 1))]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_206 = 0; i_206 < 2; ++i_206) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_206 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), __pack_half2(((bfloat16_t)scores_10[(i_206 * 8)]), ((bfloat16_t)scores_10[((i_206 * 8) + 1)])), __pack_half2(((bfloat16_t)scores_10[((i_206 * 8) + 2)]), ((bfloat16_t)scores_10[((i_206 * 8) + 3)])), __pack_half2(((bfloat16_t)scores_10[((i_206 * 8) + 4)]), ((bfloat16_t)scores_10[((i_206 * 8) + 5)])), __pack_half2(((bfloat16_t)scores_10[((i_206 * 8) + 6)]), ((bfloat16_t)scores_10[((i_206 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_21[16];
            bfloat16_t B_local_21[16];
            __syncthreads();
            for (int ki_21 = 0; ki_21 < 4; ++ki_21) {
              for (int i_207 = 0; i_207 < 2; ++i_207) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_207 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_21 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_21[(i_207 * 8)])));
              }
              for (int i_208 = 0; i_208 < 2; ++i_208) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_21 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_208) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_21[(i_208 * 8)])));
              }
              for (int i_209 = 0; i_209 < 2; ++i_209) {
                for (int j_10 = 0; j_10 < 2; ++j_10) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_10 + ((i_209 * 16) + (j_10 * 8))), reinterpret_cast<const unsigned*>(A_local_21 + (i_209 * 8)), reinterpret_cast<const unsigned*>(B_local_21 + (j_10 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_10 + (((i_209 * 16) + (j_10 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_21 + (i_209 * 8)), reinterpret_cast<const unsigned*>(B_local_21 + ((j_10 * 8) + 4)));
                }
              }
            }
          }
        }
        #pragma unroll
        for (int i_210 = 0; i_210 < 32; ++i_210) {
          if (i_210 < 8) {
            cu[i_210] = out_10[(((i_210 >> 1) * 4) + (i_210 & 1))];
          }
        }
        #pragma unroll
        for (int i_211 = 0; i_211 < 8; ++i_211) {
          cu[(i_211 + 8)] = out_10[((((i_211 >> 1) * 4) + (i_211 & 1)) + 2)];
        }
        #pragma unroll
        for (int i_212 = 0; i_212 < 8; ++i_212) {
          cu[(i_212 + 24)] = out_10[((((i_212 >> 1) * 4) + (i_212 & 1)) + 16)];
        }
      }
      __syncthreads();
      if (begin_10 < end_10) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_213 = 0; i_213 < 4; ++i_213) {
            if (i_213 < 1) {
              ((float*)buf_dyn_shmem)[((i_213 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_10[i_213];
              ((float*)buf_dyn_shmem)[(((i_213 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_10[i_213];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)] = maximum_10[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)] = denominator_10[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)] = maximum_10[2];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)] = denominator_10[2];
        }
      }
    }
    __syncthreads();
    if (label == 12) {
      int begin_11 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_11 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_11 < end_11) {
        #pragma unroll
        for (int i_214 = 0; i_214 < 2; ++i_214) {
          bfloat16_t broadcast_var_78 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_105;
          if ((i_214 < 1)) {
            condval_105 = ((((int)threadIdx.x) >> 4) + 16);
          } else {
            condval_105 = (((i_214 * 8) + (((int)threadIdx.x) >> 4)) + 16);
          }
          uint4 condval_104;
          if ((0 <= condval_105)) {
            int64_t condval_106;
            if ((((int64_t)i_214) < (int64_t)1)) {
              condval_106 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)16);
            } else {
              condval_106 = (((((int64_t)i_214) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)16);
            }
            int64_t condval_107;
            if ((((int64_t)i_214) < (int64_t)1)) {
              condval_107 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)16);
            } else {
              condval_107 = (((((int64_t)i_214) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)16);
            }
            condval_104 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_107, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_104 = make_uint4(__pack_nv_bfloat162(broadcast_var_78, broadcast_var_78), __pack_nv_bfloat162(broadcast_var_78, broadcast_var_78), __pack_nv_bfloat162(broadcast_var_78, broadcast_var_78), __pack_nv_bfloat162(broadcast_var_78, broadcast_var_78));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_214 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_104;
        }
        #pragma unroll
        for (int i_215 = 0; i_215 < 4; ++i_215) {
          float broadcast_var_79 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_11 + (i_215 * 4)) = make_float4(broadcast_var_79, broadcast_var_79, broadcast_var_79, broadcast_var_79);
        }
        #pragma unroll
        for (int i_216 = 0; i_216 < 16; ++i_216) {
          if ((i_216 & 3) < 2) {
            out_11[i_216] = cu[((((i_216 >> 2) * 2) + (i_216 & 1)) + 16)];
          }
        }
        #pragma unroll
        for (int i_217 = 0; i_217 < 8; ++i_217) {
          out_11[((((i_217 >> 1) * 4) + (i_217 & 1)) + 2)] = cu[(i_217 + 24)];
        }
        float broadcast_var_80 = -CUDART_INF_F;
        *(float2*)(maximum_11 + 0) = make_float2(broadcast_var_80, broadcast_var_80);
        float broadcast_var_81 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(denominator_11 + 0) = make_float2(broadcast_var_81, broadcast_var_81);
        #pragma unroll
        for (int i_218 = 0; i_218 < 2; ++i_218) {
          if (i_218 < 1) {
            maximum_11[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
            denominator_11[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
          }
        }
        maximum_11[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)];
        denominator_11[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)];
        for (int tile_11 = 0; tile_11 < (((end_11 + 63) - begin_11) >> 6); ++tile_11) {
          __syncthreads();
          #pragma unroll
          for (int i_219 = 0; i_219 < 8; ++i_219) {
            int condval_108;
            if ((0 <= ((((tile_11 * 64) + (i_219 * 8)) + (((int)threadIdx.x) >> 4)) + begin_11))) {
              condval_108 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_11 * 64) + (i_219 * 8)) + (((int)threadIdx.x) >> 4)) + begin_11), 173665))];
            } else {
              condval_108 = 0;
            }
            int token_11 = condval_108;
            bfloat16_t broadcast_var_82 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_109;
            if ((((((((tile_11 * 64) + (i_219 * 8)) + (((int)threadIdx.x) >> 4)) + begin_11) < end_11) && (0 <= token_11)) && (token_11 < 173666))) {
              condval_109 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_11) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_109 = make_uint4(__pack_nv_bfloat162(broadcast_var_82, broadcast_var_82), __pack_nv_bfloat162(broadcast_var_82, broadcast_var_82), __pack_nv_bfloat162(broadcast_var_82, broadcast_var_82), __pack_nv_bfloat162(broadcast_var_82, broadcast_var_82));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_219 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_109;
            bfloat16_t broadcast_var_83 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_110;
            if ((((((((tile_11 * 64) + (i_219 * 8)) + (((int)threadIdx.x) >> 4)) + begin_11) < end_11) && (0 <= token_11)) && (token_11 < 173666))) {
              condval_110 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_11) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_110 = make_uint4(__pack_nv_bfloat162(broadcast_var_83, broadcast_var_83), __pack_nv_bfloat162(broadcast_var_83, broadcast_var_83), __pack_nv_bfloat162(broadcast_var_83, broadcast_var_83), __pack_nv_bfloat162(broadcast_var_83, broadcast_var_83));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_219 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_110;
          }
          #pragma unroll
          for (int i_220 = 0; i_220 < 2; ++i_220) {
            float broadcast_var_84 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_11 + (i_220 * 4)) = make_float4(broadcast_var_84, broadcast_var_84, broadcast_var_84, broadcast_var_84);
          }
          {
            bfloat16_t A_local_22[8];
            bfloat16_t B_local_22[8];
            __syncthreads();
            for (int ki_22 = 0; ki_22 < 8; ++ki_22) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_22 >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_22[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_22 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_22[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_11 + 0), reinterpret_cast<const unsigned*>(A_local_22 + 0), reinterpret_cast<const unsigned*>(B_local_22 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_11 + 4), reinterpret_cast<const unsigned*>(A_local_22 + 0), reinterpret_cast<const unsigned*>(B_local_22 + 4));
            }
          }
          #pragma unroll
          for (int i_221 = 0; i_221 < 8; ++i_221) {
            float condval_111;
            if ((((((((tile_11 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_221 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_11) + (i_221 & 1)) < end_11)) {
              condval_111 = (scores_11[i_221] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_111 = -CUDART_INF_F;
            }
            scores_11[i_221] = condval_111;
          }
          *(float2*)(previous_11 + 0) = *(float2*)(maximum_11 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_222 = 0; i_222 < 2; ++i_222) {
            maximum_clear_11[i_222] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_22 = 0; rv_22 < 4; ++rv_22) {
              maximum_clear_11[i_222] = max(maximum_clear_11[i_222], scores_11[((((rv_22 & 1) * 4) + (i_222 * 2)) + (rv_22 >> 1))]);
            }
            maximum_clear_11[i_222] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_11[i_222], (&(((float*)buf_dyn_shmem)[13632])));
            maximum_clear_11[i_222] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_11[i_222]);
            maximum_11[i_222] = max(maximum_11[i_222], maximum_clear_11[i_222]);
          }
          #pragma unroll
          for (int i_223 = 0; i_223 < 2; ++i_223) {
            float condval_112;
            if ((previous_11[i_223] == -CUDART_INF_F)) {
              condval_112 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_112 = exp2f(((previous_11[i_223] - maximum_11[i_223]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_11[i_223] = condval_112;
          }
          #pragma unroll
          for (int i_224 = 0; i_224 < 8; ++i_224) {
            float condval_113;
            if ((scores_11[i_224] == -CUDART_INF_F)) {
              condval_113 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_113 = exp2f(((scores_11[i_224] - maximum_11[((i_224 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_11[i_224] = condval_113;
          }
          __syncthreads();
          #pragma unroll
          for (int i_225 = 0; i_225 < 2; ++i_225) {
            subtotal_11[i_225] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_23 = 0; rv_23 < 4; ++rv_23) {
              subtotal_11[i_225] = (subtotal_11[i_225] + scores_11[((((rv_23 & 1) * 4) + (i_225 * 2)) + (rv_23 >> 1))]);
            }
            subtotal_11[i_225] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_11[i_225], (&(((float*)buf_dyn_shmem)[14656])));
            subtotal_11[i_225] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_11[i_225]);
          }
          #pragma unroll
          for (int i_226 = 0; i_226 < 2; ++i_226) {
            denominator_11[i_226] = ((denominator_11[i_226] * factor_11[i_226]) + subtotal_11[i_226]);
          }
          #pragma unroll
          for (int i_227 = 0; i_227 < 16; ++i_227) {
            out_11[i_227] = (out_11[i_227] * factor_11[((i_227 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_228 = 0; i_228 < 4; ++i_228) {
            uint1 __10;
            float2 v__9 = *(float2*)(scores_11 + (i_228 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__10))[0] = __float22bfloat162_rn(((float2*)(&v__9))[0]);
            *(uint1*)(probs_local_cast_9 + 0) = __10;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_228 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_228 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_228 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_9 + 0);
          }
          {
            bfloat16_t A_local_23[8];
            bfloat16_t B_local_23[16];
            __syncthreads();
            for (int ki_23 = 0; ki_23 < 4; ++ki_23) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_23 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_23[0])));
              for (int i_229 = 0; i_229 < 2; ++i_229) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_23 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_229) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_23[(i_229 * 8)])));
              }
              for (int j_11 = 0; j_11 < 2; ++j_11) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_11 + (j_11 * 8)), reinterpret_cast<const unsigned*>(A_local_23 + 0), reinterpret_cast<const unsigned*>(B_local_23 + (j_11 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_11 + ((j_11 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_23 + 0), reinterpret_cast<const unsigned*>(B_local_23 + ((j_11 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_230 = 0; i_230 < 8; ++i_230) {
          cu[(i_230 + 16)] = out_11[(((i_230 >> 1) * 4) + (i_230 & 1))];
        }
        #pragma unroll
        for (int i_231 = 0; i_231 < 8; ++i_231) {
          cu[(i_231 + 24)] = out_11[((((i_231 >> 1) * 4) + (i_231 & 1)) + 2)];
        }
      }
      __syncthreads();
      if (begin_11 < end_11) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_232 = 0; i_232 < 2; ++i_232) {
            if (i_232 < 1) {
              ((float*)buf_dyn_shmem)[(((i_232 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 16)] = maximum_11[i_232];
              ((float*)buf_dyn_shmem)[(((i_232 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 48)] = denominator_11[i_232];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)] = maximum_11[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)] = denominator_11[1];
        }
      }
    }
    __syncthreads();
    if (label == 13) {
      int begin_12 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_12 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_12 < end_12) {
        #pragma unroll
        for (int i_233 = 0; i_233 < 4; ++i_233) {
          bfloat16_t broadcast_var_85 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_115;
          if ((i_233 < 1)) {
            condval_115 = (((int)threadIdx.x) >> 4);
          } else {
            int condval_116;
            if ((i_233 < 2)) {
              condval_116 = (((i_233 * 8) + (((int)threadIdx.x) >> 4)) + 8);
            } else {
              condval_116 = (((i_233 * 8) + (((int)threadIdx.x) >> 4)) + 8);
            }
            condval_115 = condval_116;
          }
          uint4 condval_114;
          if (((i_233 < 3) && (0 <= condval_115))) {
            int64_t condval_117;
            if ((((int64_t)i_233) < (int64_t)1)) {
              condval_117 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              int64_t condval_118;
              if ((((int64_t)i_233) < (int64_t)2)) {
                condval_118 = (((((int64_t)i_233) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              } else {
                condval_118 = (((((int64_t)i_233) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              }
              condval_117 = condval_118;
            }
            int64_t condval_119;
            if ((((int64_t)i_233) < (int64_t)1)) {
              condval_119 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              int64_t condval_120;
              if ((((int64_t)i_233) < (int64_t)2)) {
                condval_120 = (((((int64_t)i_233) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              } else {
                condval_120 = (((((int64_t)i_233) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              }
              condval_119 = condval_120;
            }
            condval_114 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_119, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_114 = make_uint4(__pack_nv_bfloat162(broadcast_var_85, broadcast_var_85), __pack_nv_bfloat162(broadcast_var_85, broadcast_var_85), __pack_nv_bfloat162(broadcast_var_85, broadcast_var_85), __pack_nv_bfloat162(broadcast_var_85, broadcast_var_85));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_233 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_114;
        }
        #pragma unroll
        for (int i_234 = 0; i_234 < 8; ++i_234) {
          float broadcast_var_86 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_12 + (i_234 * 4)) = make_float4(broadcast_var_86, broadcast_var_86, broadcast_var_86, broadcast_var_86);
        }
        #pragma unroll
        for (int i_235 = 0; i_235 < 32; ++i_235) {
          if ((((i_235 >> 4) * 2) + ((i_235 & 3) >> 1)) < 1) {
            out_12[i_235] = cu[(((((i_235 >> 4) * 16) + (((i_235 & 3) >> 1) * 8)) + (((i_235 & 15) >> 2) * 2)) + (i_235 & 1))];
          }
        }
        #pragma unroll
        for (int i_236 = 0; i_236 < 8; ++i_236) {
          out_12[((((i_236 >> 1) * 4) + (i_236 & 1)) + 2)] = cu[(i_236 + 16)];
        }
        #pragma unroll
        for (int i_237 = 0; i_237 < 8; ++i_237) {
          out_12[((((i_237 >> 1) * 4) + (i_237 & 1)) + 16)] = cu[(i_237 + 24)];
        }
        float broadcast_var_87 = -CUDART_INF_F;
        *(float4*)(maximum_12 + 0) = make_float4(broadcast_var_87, broadcast_var_87, broadcast_var_87, broadcast_var_87);
        float broadcast_var_88 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(denominator_12 + 0) = make_float4(broadcast_var_88, broadcast_var_88, broadcast_var_88, broadcast_var_88);
        #pragma unroll
        for (int i_238 = 0; i_238 < 4; ++i_238) {
          if (i_238 < 1) {
            maximum_12[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
            denominator_12[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
          }
        }
        maximum_12[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
        denominator_12[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
        maximum_12[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)];
        denominator_12[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)];
        for (int tile_12 = 0; tile_12 < (((end_12 + 63) - begin_12) >> 6); ++tile_12) {
          __syncthreads();
          #pragma unroll
          for (int i_239 = 0; i_239 < 8; ++i_239) {
            int condval_121;
            if ((0 <= ((((tile_12 * 64) + (i_239 * 8)) + (((int)threadIdx.x) >> 4)) + begin_12))) {
              condval_121 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_12 * 64) + (i_239 * 8)) + (((int)threadIdx.x) >> 4)) + begin_12), 173665))];
            } else {
              condval_121 = 0;
            }
            int token_12 = condval_121;
            bfloat16_t broadcast_var_89 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_122;
            if ((((((((tile_12 * 64) + (i_239 * 8)) + (((int)threadIdx.x) >> 4)) + begin_12) < end_12) && (0 <= token_12)) && (token_12 < 173666))) {
              condval_122 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_12) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_122 = make_uint4(__pack_nv_bfloat162(broadcast_var_89, broadcast_var_89), __pack_nv_bfloat162(broadcast_var_89, broadcast_var_89), __pack_nv_bfloat162(broadcast_var_89, broadcast_var_89), __pack_nv_bfloat162(broadcast_var_89, broadcast_var_89));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_239 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_122;
            bfloat16_t broadcast_var_90 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_123;
            if ((((((((tile_12 * 64) + (i_239 * 8)) + (((int)threadIdx.x) >> 4)) + begin_12) < end_12) && (0 <= token_12)) && (token_12 < 173666))) {
              condval_123 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_12) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_123 = make_uint4(__pack_nv_bfloat162(broadcast_var_90, broadcast_var_90), __pack_nv_bfloat162(broadcast_var_90, broadcast_var_90), __pack_nv_bfloat162(broadcast_var_90, broadcast_var_90), __pack_nv_bfloat162(broadcast_var_90, broadcast_var_90));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_239 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_123;
          }
          #pragma unroll
          for (int i_240 = 0; i_240 < 4; ++i_240) {
            float broadcast_var_91 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_12 + (i_240 * 4)) = make_float4(broadcast_var_91, broadcast_var_91, broadcast_var_91, broadcast_var_91);
          }
          {
            bfloat16_t A_local_24[16];
            bfloat16_t B_local_24[8];
            __syncthreads();
            for (int ki_24 = 0; ki_24 < 8; ++ki_24) {
              for (int i_241 = 0; i_241 < 2; ++i_241) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_24 >> 2) * 2048) + (i_241 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_24 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_24 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_24[(i_241 * 8)])));
              }
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_24 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_24 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_24 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_24[0])));
              for (int i_242 = 0; i_242 < 2; ++i_242) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_12 + (i_242 * 8)), reinterpret_cast<const unsigned*>(A_local_24 + (i_242 * 8)), reinterpret_cast<const unsigned*>(B_local_24 + 0));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_12 + ((i_242 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_24 + (i_242 * 8)), reinterpret_cast<const unsigned*>(B_local_24 + 4));
              }
            }
          }
          #pragma unroll
          for (int i_243 = 0; i_243 < 16; ++i_243) {
            float condval_124;
            if ((((((i_243 >> 3) * 2) + ((i_243 & 3) >> 1)) < 3) && (((((((tile_12 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_243 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_12) + (i_243 & 1)) < end_12))) {
              condval_124 = (scores_12[i_243] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_124 = -CUDART_INF_F;
            }
            scores_12[i_243] = condval_124;
          }
          *(float4*)(previous_12 + 0) = *(float4*)(maximum_12 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_244 = 0; i_244 < 4; ++i_244) {
            maximum_clear_12[i_244] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_24 = 0; rv_24 < 4; ++rv_24) {
              maximum_clear_12[i_244] = max(maximum_clear_12[i_244], scores_12[(((((i_244 >> 1) * 8) + ((rv_24 & 1) * 4)) + ((i_244 & 1) * 2)) + (rv_24 >> 1))]);
            }
            maximum_clear_12[i_244] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_12[i_244], (&(((float*)buf_dyn_shmem)[11456])));
            maximum_clear_12[i_244] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_12[i_244]);
            maximum_12[i_244] = max(maximum_12[i_244], maximum_clear_12[i_244]);
          }
          #pragma unroll
          for (int i_245 = 0; i_245 < 4; ++i_245) {
            float condval_125;
            if ((previous_12[i_245] == -CUDART_INF_F)) {
              condval_125 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_125 = exp2f(((previous_12[i_245] - maximum_12[i_245]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_12[i_245] = condval_125;
          }
          #pragma unroll
          for (int i_246 = 0; i_246 < 16; ++i_246) {
            float condval_126;
            if ((scores_12[i_246] == -CUDART_INF_F)) {
              condval_126 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_126 = exp2f(((scores_12[i_246] - maximum_12[(((i_246 >> 3) * 2) + ((i_246 & 3) >> 1))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_12[i_246] = condval_126;
          }
          __syncthreads();
          #pragma unroll
          for (int i_247 = 0; i_247 < 4; ++i_247) {
            subtotal_12[i_247] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_25 = 0; rv_25 < 4; ++rv_25) {
              subtotal_12[i_247] = (subtotal_12[i_247] + scores_12[(((((i_247 >> 1) * 8) + ((rv_25 & 1) * 4)) + ((i_247 & 1) * 2)) + (rv_25 >> 1))]);
            }
            subtotal_12[i_247] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_12[i_247], (&(((float*)buf_dyn_shmem)[14784])));
            subtotal_12[i_247] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_12[i_247]);
          }
          #pragma unroll
          for (int i_248 = 0; i_248 < 4; ++i_248) {
            denominator_12[i_248] = ((denominator_12[i_248] * factor_12[i_248]) + subtotal_12[i_248]);
          }
          #pragma unroll
          for (int i_249 = 0; i_249 < 32; ++i_249) {
            out_12[i_249] = (out_12[i_249] * factor_12[(((i_249 >> 4) * 2) + ((i_249 & 3) >> 1))]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_250 = 0; i_250 < 2; ++i_250) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_250 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), __pack_half2(((bfloat16_t)scores_12[(i_250 * 8)]), ((bfloat16_t)scores_12[((i_250 * 8) + 1)])), __pack_half2(((bfloat16_t)scores_12[((i_250 * 8) + 2)]), ((bfloat16_t)scores_12[((i_250 * 8) + 3)])), __pack_half2(((bfloat16_t)scores_12[((i_250 * 8) + 4)]), ((bfloat16_t)scores_12[((i_250 * 8) + 5)])), __pack_half2(((bfloat16_t)scores_12[((i_250 * 8) + 6)]), ((bfloat16_t)scores_12[((i_250 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_25[16];
            bfloat16_t B_local_25[16];
            __syncthreads();
            for (int ki_25 = 0; ki_25 < 4; ++ki_25) {
              for (int i_251 = 0; i_251 < 2; ++i_251) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_251 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_25 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_25 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_25[(i_251 * 8)])));
              }
              for (int i_252 = 0; i_252 < 2; ++i_252) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_25 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_252) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_25[(i_252 * 8)])));
              }
              for (int i_253 = 0; i_253 < 2; ++i_253) {
                for (int j_12 = 0; j_12 < 2; ++j_12) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_12 + ((i_253 * 16) + (j_12 * 8))), reinterpret_cast<const unsigned*>(A_local_25 + (i_253 * 8)), reinterpret_cast<const unsigned*>(B_local_25 + (j_12 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_12 + (((i_253 * 16) + (j_12 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_25 + (i_253 * 8)), reinterpret_cast<const unsigned*>(B_local_25 + ((j_12 * 8) + 4)));
                }
              }
            }
          }
        }
        #pragma unroll
        for (int i_254 = 0; i_254 < 32; ++i_254) {
          if (i_254 < 8) {
            cu[i_254] = out_12[(((i_254 >> 1) * 4) + (i_254 & 1))];
          }
        }
        #pragma unroll
        for (int i_255 = 0; i_255 < 8; ++i_255) {
          cu[(i_255 + 16)] = out_12[((((i_255 >> 1) * 4) + (i_255 & 1)) + 2)];
        }
        #pragma unroll
        for (int i_256 = 0; i_256 < 8; ++i_256) {
          cu[(i_256 + 24)] = out_12[((((i_256 >> 1) * 4) + (i_256 & 1)) + 16)];
        }
      }
      __syncthreads();
      if (begin_12 < end_12) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_257 = 0; i_257 < 4; ++i_257) {
            if (i_257 < 1) {
              ((float*)buf_dyn_shmem)[((i_257 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_12[i_257];
              ((float*)buf_dyn_shmem)[(((i_257 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_12[i_257];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_12[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_12[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)] = maximum_12[2];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)] = denominator_12[2];
        }
      }
    }
    __syncthreads();
    if (label == 14) {
      int begin_13 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_13 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_13 < end_13) {
        #pragma unroll
        for (int i_258 = 0; i_258 < 4; ++i_258) {
          bfloat16_t broadcast_var_92 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_128;
          if ((i_258 < 1)) {
            condval_128 = ((((int)threadIdx.x) >> 4) + 8);
          } else {
            int condval_129;
            if ((i_258 < 2)) {
              condval_129 = (((i_258 * 8) + (((int)threadIdx.x) >> 4)) + 8);
            } else {
              condval_129 = (((i_258 * 8) + (((int)threadIdx.x) >> 4)) + 8);
            }
            condval_128 = condval_129;
          }
          uint4 condval_127;
          if (((i_258 < 3) && (0 <= condval_128))) {
            int64_t condval_130;
            if ((((int64_t)i_258) < (int64_t)1)) {
              condval_130 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)8);
            } else {
              int64_t condval_131;
              if ((((int64_t)i_258) < (int64_t)2)) {
                condval_131 = (((((int64_t)i_258) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              } else {
                condval_131 = (((((int64_t)i_258) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              }
              condval_130 = condval_131;
            }
            int64_t condval_132;
            if ((((int64_t)i_258) < (int64_t)1)) {
              condval_132 = ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (int64_t)8);
            } else {
              int64_t condval_133;
              if ((((int64_t)i_258) < (int64_t)2)) {
                condval_133 = (((((int64_t)i_258) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              } else {
                condval_133 = (((((int64_t)i_258) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (int64_t)8);
              }
              condval_132 = condval_133;
            }
            condval_127 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_132, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_127 = make_uint4(__pack_nv_bfloat162(broadcast_var_92, broadcast_var_92), __pack_nv_bfloat162(broadcast_var_92, broadcast_var_92), __pack_nv_bfloat162(broadcast_var_92, broadcast_var_92), __pack_nv_bfloat162(broadcast_var_92, broadcast_var_92));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_258 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_127;
        }
        #pragma unroll
        for (int i_259 = 0; i_259 < 8; ++i_259) {
          float broadcast_var_93 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_13 + (i_259 * 4)) = make_float4(broadcast_var_93, broadcast_var_93, broadcast_var_93, broadcast_var_93);
        }
        #pragma unroll
        for (int i_260 = 0; i_260 < 32; ++i_260) {
          if ((((i_260 >> 4) * 2) + ((i_260 & 3) >> 1)) < 1) {
            out_13[i_260] = cu[((((((i_260 >> 4) * 16) + (((i_260 & 3) >> 1) * 8)) + (((i_260 & 15) >> 2) * 2)) + (i_260 & 1)) + 8)];
          }
        }
        #pragma unroll
        for (int i_261 = 0; i_261 < 8; ++i_261) {
          out_13[((((i_261 >> 1) * 4) + (i_261 & 1)) + 2)] = cu[(i_261 + 16)];
        }
        #pragma unroll
        for (int i_262 = 0; i_262 < 8; ++i_262) {
          out_13[((((i_262 >> 1) * 4) + (i_262 & 1)) + 16)] = cu[(i_262 + 24)];
        }
        float broadcast_var_94 = -CUDART_INF_F;
        *(float4*)(maximum_13 + 0) = make_float4(broadcast_var_94, broadcast_var_94, broadcast_var_94, broadcast_var_94);
        float broadcast_var_95 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(denominator_13 + 0) = make_float4(broadcast_var_95, broadcast_var_95, broadcast_var_95, broadcast_var_95);
        #pragma unroll
        for (int i_263 = 0; i_263 < 4; ++i_263) {
          if (i_263 < 1) {
            maximum_13[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
            denominator_13[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
          }
        }
        maximum_13[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
        denominator_13[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
        maximum_13[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)];
        denominator_13[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)];
        for (int tile_13 = 0; tile_13 < (((end_13 + 63) - begin_13) >> 6); ++tile_13) {
          __syncthreads();
          #pragma unroll
          for (int i_264 = 0; i_264 < 8; ++i_264) {
            int condval_134;
            if ((0 <= ((((tile_13 * 64) + (i_264 * 8)) + (((int)threadIdx.x) >> 4)) + begin_13))) {
              condval_134 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_13 * 64) + (i_264 * 8)) + (((int)threadIdx.x) >> 4)) + begin_13), 173665))];
            } else {
              condval_134 = 0;
            }
            int token_13 = condval_134;
            bfloat16_t broadcast_var_96 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_135;
            if ((((((((tile_13 * 64) + (i_264 * 8)) + (((int)threadIdx.x) >> 4)) + begin_13) < end_13) && (0 <= token_13)) && (token_13 < 173666))) {
              condval_135 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_13) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_135 = make_uint4(__pack_nv_bfloat162(broadcast_var_96, broadcast_var_96), __pack_nv_bfloat162(broadcast_var_96, broadcast_var_96), __pack_nv_bfloat162(broadcast_var_96, broadcast_var_96), __pack_nv_bfloat162(broadcast_var_96, broadcast_var_96));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_264 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_135;
            bfloat16_t broadcast_var_97 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_136;
            if ((((((((tile_13 * 64) + (i_264 * 8)) + (((int)threadIdx.x) >> 4)) + begin_13) < end_13) && (0 <= token_13)) && (token_13 < 173666))) {
              condval_136 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_13) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_136 = make_uint4(__pack_nv_bfloat162(broadcast_var_97, broadcast_var_97), __pack_nv_bfloat162(broadcast_var_97, broadcast_var_97), __pack_nv_bfloat162(broadcast_var_97, broadcast_var_97), __pack_nv_bfloat162(broadcast_var_97, broadcast_var_97));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_264 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_136;
          }
          #pragma unroll
          for (int i_265 = 0; i_265 < 4; ++i_265) {
            float broadcast_var_98 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_13 + (i_265 * 4)) = make_float4(broadcast_var_98, broadcast_var_98, broadcast_var_98, broadcast_var_98);
          }
          {
            bfloat16_t A_local_26[16];
            bfloat16_t B_local_26[8];
            __syncthreads();
            for (int ki_26 = 0; ki_26 < 8; ++ki_26) {
              for (int i_266 = 0; i_266 < 2; ++i_266) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_26 >> 2) * 2048) + (i_266 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_26 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_26 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_26[(i_266 * 8)])));
              }
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_26 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_26 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_26 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_26[0])));
              for (int i_267 = 0; i_267 < 2; ++i_267) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_13 + (i_267 * 8)), reinterpret_cast<const unsigned*>(A_local_26 + (i_267 * 8)), reinterpret_cast<const unsigned*>(B_local_26 + 0));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_13 + ((i_267 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_26 + (i_267 * 8)), reinterpret_cast<const unsigned*>(B_local_26 + 4));
              }
            }
          }
          #pragma unroll
          for (int i_268 = 0; i_268 < 16; ++i_268) {
            float condval_137;
            if ((((((i_268 >> 3) * 2) + ((i_268 & 3) >> 1)) < 3) && (((((((tile_13 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_268 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_13) + (i_268 & 1)) < end_13))) {
              condval_137 = (scores_13[i_268] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_137 = -CUDART_INF_F;
            }
            scores_13[i_268] = condval_137;
          }
          *(float4*)(previous_13 + 0) = *(float4*)(maximum_13 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_269 = 0; i_269 < 4; ++i_269) {
            maximum_clear_13[i_269] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_26 = 0; rv_26 < 4; ++rv_26) {
              maximum_clear_13[i_269] = max(maximum_clear_13[i_269], scores_13[(((((i_269 >> 1) * 8) + ((rv_26 & 1) * 4)) + ((i_269 & 1) * 2)) + (rv_26 >> 1))]);
            }
            maximum_clear_13[i_269] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_13[i_269], (&(((float*)buf_dyn_shmem)[15040])));
            maximum_clear_13[i_269] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_13[i_269]);
            maximum_13[i_269] = max(maximum_13[i_269], maximum_clear_13[i_269]);
          }
          #pragma unroll
          for (int i_270 = 0; i_270 < 4; ++i_270) {
            float condval_138;
            if ((previous_13[i_270] == -CUDART_INF_F)) {
              condval_138 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_138 = exp2f(((previous_13[i_270] - maximum_13[i_270]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_13[i_270] = condval_138;
          }
          #pragma unroll
          for (int i_271 = 0; i_271 < 16; ++i_271) {
            float condval_139;
            if ((scores_13[i_271] == -CUDART_INF_F)) {
              condval_139 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_139 = exp2f(((scores_13[i_271] - maximum_13[(((i_271 >> 3) * 2) + ((i_271 & 3) >> 1))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_13[i_271] = condval_139;
          }
          __syncthreads();
          #pragma unroll
          for (int i_272 = 0; i_272 < 4; ++i_272) {
            subtotal_13[i_272] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_27 = 0; rv_27 < 4; ++rv_27) {
              subtotal_13[i_272] = (subtotal_13[i_272] + scores_13[(((((i_272 >> 1) * 8) + ((rv_27 & 1) * 4)) + ((i_272 & 1) * 2)) + (rv_27 >> 1))]);
            }
            subtotal_13[i_272] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_13[i_272], (&(((float*)buf_dyn_shmem)[14144])));
            subtotal_13[i_272] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_13[i_272]);
          }
          #pragma unroll
          for (int i_273 = 0; i_273 < 4; ++i_273) {
            denominator_13[i_273] = ((denominator_13[i_273] * factor_13[i_273]) + subtotal_13[i_273]);
          }
          #pragma unroll
          for (int i_274 = 0; i_274 < 32; ++i_274) {
            out_13[i_274] = (out_13[i_274] * factor_13[(((i_274 >> 4) * 2) + ((i_274 & 3) >> 1))]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_275 = 0; i_275 < 2; ++i_275) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_275 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), __pack_half2(((bfloat16_t)scores_13[(i_275 * 8)]), ((bfloat16_t)scores_13[((i_275 * 8) + 1)])), __pack_half2(((bfloat16_t)scores_13[((i_275 * 8) + 2)]), ((bfloat16_t)scores_13[((i_275 * 8) + 3)])), __pack_half2(((bfloat16_t)scores_13[((i_275 * 8) + 4)]), ((bfloat16_t)scores_13[((i_275 * 8) + 5)])), __pack_half2(((bfloat16_t)scores_13[((i_275 * 8) + 6)]), ((bfloat16_t)scores_13[((i_275 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_27[16];
            bfloat16_t B_local_27[16];
            __syncthreads();
            for (int ki_27 = 0; ki_27 < 4; ++ki_27) {
              for (int i_276 = 0; i_276 < 2; ++i_276) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_276 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_27 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_27[(i_276 * 8)])));
              }
              for (int i_277 = 0; i_277 < 2; ++i_277) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_27 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_277) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_27[(i_277 * 8)])));
              }
              for (int i_278 = 0; i_278 < 2; ++i_278) {
                for (int j_13 = 0; j_13 < 2; ++j_13) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_13 + ((i_278 * 16) + (j_13 * 8))), reinterpret_cast<const unsigned*>(A_local_27 + (i_278 * 8)), reinterpret_cast<const unsigned*>(B_local_27 + (j_13 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_13 + (((i_278 * 16) + (j_13 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_27 + (i_278 * 8)), reinterpret_cast<const unsigned*>(B_local_27 + ((j_13 * 8) + 4)));
                }
              }
            }
          }
        }
        #pragma unroll
        for (int i_279 = 0; i_279 < 8; ++i_279) {
          cu[(i_279 + 8)] = out_13[(((i_279 >> 1) * 4) + (i_279 & 1))];
        }
        #pragma unroll
        for (int i_280 = 0; i_280 < 8; ++i_280) {
          cu[(i_280 + 16)] = out_13[((((i_280 >> 1) * 4) + (i_280 & 1)) + 2)];
        }
        #pragma unroll
        for (int i_281 = 0; i_281 < 8; ++i_281) {
          cu[(i_281 + 24)] = out_13[((((i_281 >> 1) * 4) + (i_281 & 1)) + 16)];
        }
      }
      __syncthreads();
      if (begin_13 < end_13) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_282 = 0; i_282 < 4; ++i_282) {
            if (i_282 < 1) {
              ((float*)buf_dyn_shmem)[(((i_282 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 8)] = maximum_13[i_282];
              ((float*)buf_dyn_shmem)[(((i_282 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 40)] = denominator_13[i_282];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_13[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_13[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)] = maximum_13[2];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)] = denominator_13[2];
        }
      }
    }
    __syncthreads();
    if (label == 15) {
      int begin_14 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      int end_14 = ((((Offsets[(((((int)blockIdx.x) * 16) + r) + 1)] - Offsets[((((int)blockIdx.x) * 16) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 16) + r)]);
      if (begin_14 < end_14) {
        #pragma unroll
        for (int i_283 = 0; i_283 < 4; ++i_283) {
          bfloat16_t broadcast_var_99 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_141;
          if ((i_283 < 1)) {
            condval_141 = (((int)threadIdx.x) >> 4);
          } else {
            int condval_142;
            if ((i_283 < 2)) {
              condval_142 = ((i_283 * 8) + (((int)threadIdx.x) >> 4));
            } else {
              int condval_143;
              if ((i_283 < 3)) {
                condval_143 = ((i_283 * 8) + (((int)threadIdx.x) >> 4));
              } else {
                condval_143 = ((i_283 * 8) + (((int)threadIdx.x) >> 4));
              }
              condval_142 = condval_143;
            }
            condval_141 = condval_142;
          }
          uint4 condval_140;
          if ((0 <= condval_141)) {
            int64_t condval_144;
            if ((((int64_t)i_283) < (int64_t)1)) {
              condval_144 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              int64_t condval_145;
              if ((((int64_t)i_283) < (int64_t)2)) {
                condval_145 = ((((int64_t)i_283) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
              } else {
                int64_t condval_146;
                if ((((int64_t)i_283) < (int64_t)3)) {
                  condval_146 = ((((int64_t)i_283) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
                } else {
                  condval_146 = ((((int64_t)i_283) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
                }
                condval_145 = condval_146;
              }
              condval_144 = condval_145;
            }
            int64_t condval_147;
            if ((((int64_t)i_283) < (int64_t)1)) {
              condval_147 = (((int64_t)((int)threadIdx.x)) >> (int64_t)4);
            } else {
              int64_t condval_148;
              if ((((int64_t)i_283) < (int64_t)2)) {
                condval_148 = ((((int64_t)i_283) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
              } else {
                int64_t condval_149;
                if ((((int64_t)i_283) < (int64_t)3)) {
                  condval_149 = ((((int64_t)i_283) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
                } else {
                  condval_149 = ((((int64_t)i_283) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4));
                }
                condval_148 = condval_149;
              }
              condval_147 = condval_148;
            }
            condval_140 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (min(condval_147, (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_140 = make_uint4(__pack_nv_bfloat162(broadcast_var_99, broadcast_var_99), __pack_nv_bfloat162(broadcast_var_99, broadcast_var_99), __pack_nv_bfloat162(broadcast_var_99, broadcast_var_99), __pack_nv_bfloat162(broadcast_var_99, broadcast_var_99));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_283 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_140;
        }
        #pragma unroll
        for (int i_284 = 0; i_284 < 8; ++i_284) {
          float broadcast_var_100 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_14 + (i_284 * 4)) = make_float4(broadcast_var_100, broadcast_var_100, broadcast_var_100, broadcast_var_100);
        }
        #pragma unroll
        for (int i_285 = 0; i_285 < 32; ++i_285) {
          if ((((i_285 >> 4) * 2) + ((i_285 & 3) >> 1)) < 1) {
            out_14[i_285] = cu[(((((i_285 >> 4) * 16) + (((i_285 & 3) >> 1) * 8)) + (((i_285 & 15) >> 2) * 2)) + (i_285 & 1))];
          }
        }
        #pragma unroll
        for (int i_286 = 0; i_286 < 8; ++i_286) {
          out_14[((((i_286 >> 1) * 4) + (i_286 & 1)) + 2)] = cu[(i_286 + 8)];
        }
        #pragma unroll
        for (int i_287 = 0; i_287 < 8; ++i_287) {
          out_14[((((i_287 >> 1) * 4) + (i_287 & 1)) + 16)] = cu[(i_287 + 16)];
        }
        #pragma unroll
        for (int i_288 = 0; i_288 < 8; ++i_288) {
          out_14[(((((((((int)threadIdx.x) & 31) + 96) >> 6) * 16) + ((i_288 >> 1) * 4)) + (i_288 & 1)) + 2)] = cu[(i_288 + 24)];
        }
        float broadcast_var_101 = -CUDART_INF_F;
        *(float4*)(maximum_14 + 0) = make_float4(broadcast_var_101, broadcast_var_101, broadcast_var_101, broadcast_var_101);
        float broadcast_var_102 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(denominator_14 + 0) = make_float4(broadcast_var_102, broadcast_var_102, broadcast_var_102, broadcast_var_102);
        #pragma unroll
        for (int i_289 = 0; i_289 < 4; ++i_289) {
          if (i_289 < 1) {
            maximum_14[0] = ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 31) >> 2)];
            denominator_14[0] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 32)];
          }
        }
        maximum_14[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)];
        denominator_14[1] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)];
        maximum_14[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)];
        denominator_14[2] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)];
        maximum_14[3] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)];
        denominator_14[3] = ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)];
        for (int tile_14 = 0; tile_14 < (((end_14 + 63) - begin_14) >> 6); ++tile_14) {
          __syncthreads();
          #pragma unroll
          for (int i_290 = 0; i_290 < 8; ++i_290) {
            int condval_150;
            if ((0 <= ((((tile_14 * 64) + (i_290 * 8)) + (((int)threadIdx.x) >> 4)) + begin_14))) {
              condval_150 = Indices[((((int)blockIdx.x) * 173666) + min(((((tile_14 * 64) + (i_290 * 8)) + (((int)threadIdx.x) >> 4)) + begin_14), 173665))];
            } else {
              condval_150 = 0;
            }
            int token_14 = condval_150;
            bfloat16_t broadcast_var_103 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_151;
            if ((((((((tile_14 * 64) + (i_290 * 8)) + (((int)threadIdx.x) >> 4)) + begin_14) < end_14) && (0 <= token_14)) && (token_14 < 173666))) {
              condval_151 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_14) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_151 = make_uint4(__pack_nv_bfloat162(broadcast_var_103, broadcast_var_103), __pack_nv_bfloat162(broadcast_var_103, broadcast_var_103), __pack_nv_bfloat162(broadcast_var_103, broadcast_var_103), __pack_nv_bfloat162(broadcast_var_103, broadcast_var_103));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_290 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_151;
            bfloat16_t broadcast_var_104 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_152;
            if ((((((((tile_14 * 64) + (i_290 * 8)) + (((int)threadIdx.x) >> 4)) + begin_14) < end_14) && (0 <= token_14)) && (token_14 < 173666))) {
              condval_152 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)22229248) + (((int64_t)token_14) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_152 = make_uint4(__pack_nv_bfloat162(broadcast_var_104, broadcast_var_104), __pack_nv_bfloat162(broadcast_var_104, broadcast_var_104), __pack_nv_bfloat162(broadcast_var_104, broadcast_var_104), __pack_nv_bfloat162(broadcast_var_104, broadcast_var_104));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_290 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_152;
          }
          #pragma unroll
          for (int i_291 = 0; i_291 < 4; ++i_291) {
            float broadcast_var_105 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_14 + (i_291 * 4)) = make_float4(broadcast_var_105, broadcast_var_105, broadcast_var_105, broadcast_var_105);
          }
          {
            bfloat16_t A_local_28[16];
            bfloat16_t B_local_28[8];
            __syncthreads();
            for (int ki_28 = 0; ki_28 < 8; ++ki_28) {
              for (int i_292 = 0; i_292 < 2; ++i_292) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_28 >> 2) * 2048) + (i_292 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_28 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_28[(i_292 * 8)])));
              }
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_28 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_28 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_28[0])));
              for (int i_293 = 0; i_293 < 2; ++i_293) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_14 + (i_293 * 8)), reinterpret_cast<const unsigned*>(A_local_28 + (i_293 * 8)), reinterpret_cast<const unsigned*>(B_local_28 + 0));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_14 + ((i_293 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_28 + (i_293 * 8)), reinterpret_cast<const unsigned*>(B_local_28 + 4));
              }
            }
          }
          #pragma unroll
          for (int i_294 = 0; i_294 < 16; ++i_294) {
            float condval_153;
            if ((((((((tile_14 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_294 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_14) + (i_294 & 1)) < end_14)) {
              condval_153 = (scores_14[i_294] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_153 = -CUDART_INF_F;
            }
            scores_14[i_294] = condval_153;
          }
          *(float4*)(previous_14 + 0) = *(float4*)(maximum_14 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_295 = 0; i_295 < 4; ++i_295) {
            maximum_clear_14[i_295] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_28 = 0; rv_28 < 4; ++rv_28) {
              maximum_clear_14[i_295] = max(maximum_clear_14[i_295], scores_14[(((((i_295 >> 1) * 8) + ((rv_28 & 1) * 4)) + ((i_295 & 1) * 2)) + (rv_28 >> 1))]);
            }
            maximum_clear_14[i_295] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_14[i_295], (&(((float*)buf_dyn_shmem)[11328])));
            maximum_clear_14[i_295] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_14[i_295]);
            maximum_14[i_295] = max(maximum_14[i_295], maximum_clear_14[i_295]);
          }
          #pragma unroll
          for (int i_296 = 0; i_296 < 4; ++i_296) {
            float condval_154;
            if ((previous_14[i_296] == -CUDART_INF_F)) {
              condval_154 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_154 = exp2f(((previous_14[i_296] - maximum_14[i_296]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_14[i_296] = condval_154;
          }
          #pragma unroll
          for (int i_297 = 0; i_297 < 16; ++i_297) {
            float condval_155;
            if ((scores_14[i_297] == -CUDART_INF_F)) {
              condval_155 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_155 = exp2f(((scores_14[i_297] - maximum_14[(((i_297 >> 3) * 2) + ((i_297 & 3) >> 1))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_14[i_297] = condval_155;
          }
          __syncthreads();
          #pragma unroll
          for (int i_298 = 0; i_298 < 4; ++i_298) {
            subtotal_14[i_298] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_29 = 0; rv_29 < 4; ++rv_29) {
              subtotal_14[i_298] = (subtotal_14[i_298] + scores_14[(((((i_298 >> 1) * 8) + ((rv_29 & 1) * 4)) + ((i_298 & 1) * 2)) + (rv_29 >> 1))]);
            }
            subtotal_14[i_298] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_14[i_298], (&(((float*)buf_dyn_shmem)[14912])));
            subtotal_14[i_298] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_14[i_298]);
          }
          #pragma unroll
          for (int i_299 = 0; i_299 < 4; ++i_299) {
            denominator_14[i_299] = ((denominator_14[i_299] * factor_14[i_299]) + subtotal_14[i_299]);
          }
          #pragma unroll
          for (int i_300 = 0; i_300 < 32; ++i_300) {
            out_14[i_300] = (out_14[i_300] * factor_14[(((i_300 >> 4) * 2) + ((i_300 & 3) >> 1))]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_301 = 0; i_301 < 2; ++i_301) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_301 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), __pack_half2(((bfloat16_t)scores_14[(i_301 * 8)]), ((bfloat16_t)scores_14[((i_301 * 8) + 1)])), __pack_half2(((bfloat16_t)scores_14[((i_301 * 8) + 2)]), ((bfloat16_t)scores_14[((i_301 * 8) + 3)])), __pack_half2(((bfloat16_t)scores_14[((i_301 * 8) + 4)]), ((bfloat16_t)scores_14[((i_301 * 8) + 5)])), __pack_half2(((bfloat16_t)scores_14[((i_301 * 8) + 6)]), ((bfloat16_t)scores_14[((i_301 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_29[16];
            bfloat16_t B_local_29[16];
            __syncthreads();
            for (int ki_29 = 0; ki_29 < 4; ++ki_29) {
              for (int i_302 = 0; i_302 < 2; ++i_302) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_302 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_29 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_29[(i_302 * 8)])));
              }
              for (int i_303 = 0; i_303 < 2; ++i_303) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_29 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_303) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_29[(i_303 * 8)])));
              }
              for (int i_304 = 0; i_304 < 2; ++i_304) {
                for (int j_14 = 0; j_14 < 2; ++j_14) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_14 + ((i_304 * 16) + (j_14 * 8))), reinterpret_cast<const unsigned*>(A_local_29 + (i_304 * 8)), reinterpret_cast<const unsigned*>(B_local_29 + (j_14 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_14 + (((i_304 * 16) + (j_14 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_29 + (i_304 * 8)), reinterpret_cast<const unsigned*>(B_local_29 + ((j_14 * 8) + 4)));
                }
              }
            }
          }
        }
        #pragma unroll
        for (int i_305 = 0; i_305 < 32; ++i_305) {
          if (i_305 < 8) {
            cu[i_305] = out_14[(((i_305 >> 1) * 4) + (i_305 & 1))];
          }
        }
        #pragma unroll
        for (int i_306 = 0; i_306 < 8; ++i_306) {
          cu[(i_306 + 8)] = out_14[((((i_306 >> 1) * 4) + (i_306 & 1)) + 2)];
        }
        #pragma unroll
        for (int i_307 = 0; i_307 < 8; ++i_307) {
          cu[(i_307 + 16)] = out_14[((((i_307 >> 1) * 4) + (i_307 & 1)) + 16)];
        }
        #pragma unroll
        for (int i_308 = 0; i_308 < 8; ++i_308) {
          cu[(i_308 + 24)] = out_14[(((((((((int)threadIdx.x) & 31) + 96) >> 6) * 16) + ((i_308 >> 1) * 4)) + (i_308 & 1)) + 2)];
        }
      }
      __syncthreads();
      if (begin_14 < end_14) {
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_309 = 0; i_309 < 4; ++i_309) {
            if (i_309 < 1) {
              ((float*)buf_dyn_shmem)[((i_309 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = maximum_14[i_309];
              ((float*)buf_dyn_shmem)[(((i_309 * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)] = denominator_14[i_309];
            }
          }
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 8)] = maximum_14[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 40)] = denominator_14[1];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 16)] = maximum_14[2];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 48)] = denominator_14[2];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 24)] = maximum_14[3];
          ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 31) >> 2) + 56)] = denominator_14[3];
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i_310 = 0; i_310 < 16; ++i_310) {
    *(float2*)(U + (((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 4096)) + ((i_310 >> 2) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_310 & 3) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(cu + (i_310 * 2));
  }
  if (((int)threadIdx.x) < 32) {
    Mout[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x))] = ((float*)buf_dyn_shmem)[((int)threadIdx.x)];
    Lout[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x))] = ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)];
  }
}

