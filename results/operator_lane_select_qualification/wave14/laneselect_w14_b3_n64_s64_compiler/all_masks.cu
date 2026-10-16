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

extern "C" __global__ void main_kernel(const int* __restrict__ Count, const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, const int* __restrict__ Labels, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const int* __restrict__ Owner, const bfloat16_t* __restrict__ Q, const int* __restrict__ RowMap, float* __restrict__ U, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ Count, const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, const int* __restrict__ Labels, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const int* __restrict__ Owner, const bfloat16_t* __restrict__ Q, const int* __restrict__ RowMap, float* __restrict__ U, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float cu0[8];
  float cu1[8];
  float cu2[8];
  float cu3[8];
  float out[16];
  float maximum[2];
  float denominator[2];
  float OUT0[8];
  float OUT1[8];
  float out_1[32];
  float maximum_1[4];
  float denominator_1[4];
  float OUT0_1[8];
  float OUT1_1[8];
  float OUT2[8];
  float OUT3[8];
  float scores[8];
  float previous[2];
  float maximum_clear[2];
  float factor[2];
  float subtotal[2];
  bfloat16_t probs_local_cast[2];
  float scores_1[16];
  float previous_1[4];
  float maximum_clear_1[4];
  float factor_1[4];
  float subtotal_1[4];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu0 + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu1 + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu2 + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu3 + (i_3 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  if (((int)threadIdx.x) < 32) {
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = -CUDART_INF_F;
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)] = 0x0p+0f/*0.000000e+00*/;
  }
  for (int r = 0; r < 7; ++r) {
    if (Count[((((int)blockIdx.x) * 7) + r)] <= 16) {
      int begin = ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 8) + r)]);
      int end = ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 8) + r)]);
      int label = Labels[r];
      if (begin < end) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          int src = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_4 * 8)) + (((int)threadIdx.x) >> 4))];
          bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval;
          if (((((i_4 * 8) + (((int)threadIdx.x) >> 4)) < Count[((((int)blockIdx.x) * 7) + r)]) && (0 <= src))) {
            condval = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (min(((int64_t)src), (int64_t)23) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval;
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out + (i_5 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
        }
        if (8 <= Count[((((int)blockIdx.x) * 7) + r)]) {
          int request = (RowMap[((((int)blockIdx.x) * 224) + (r * 32))] >> 3);
          #pragma unroll
          for (int i_6 = 0; i_6 < 16; ++i_6) {
            if ((i_6 & 3) < 2) {
              float condval_1;
              if ((request == 0)) {
                condval_1 = cu0[(((i_6 >> 2) * 2) + (i_6 & 1))];
              } else {
                float condval_2;
                if ((request == 1)) {
                  condval_2 = cu1[(((i_6 >> 2) * 2) + (i_6 & 1))];
                } else {
                  float condval_3;
                  if ((request == 2)) {
                    condval_3 = cu2[(((i_6 >> 2) * 2) + (i_6 & 1))];
                  } else {
                    condval_3 = cu3[(((i_6 >> 2) * 2) + (i_6 & 1))];
                  }
                  condval_2 = condval_3;
                }
                condval_1 = condval_2;
              }
              out[i_6] = condval_1;
            }
          }
        }
        if (16 <= Count[((((int)blockIdx.x) * 7) + r)]) {
          int request_1 = (RowMap[(((((int)blockIdx.x) * 224) + (r * 32)) + 8)] >> 3);
          #pragma unroll
          for (int i_7 = 0; i_7 < 8; ++i_7) {
            float condval_4;
            if ((request_1 == 0)) {
              condval_4 = cu0[i_7];
            } else {
              float condval_5;
              if ((request_1 == 1)) {
                condval_5 = cu1[i_7];
              } else {
                float condval_6;
                if ((request_1 == 2)) {
                  condval_6 = cu2[i_7];
                } else {
                  condval_6 = cu3[i_7];
                }
                condval_5 = condval_6;
              }
              condval_4 = condval_5;
            }
            out[((((i_7 >> 1) * 4) + (i_7 & 1)) + 2)] = condval_4;
          }
        }
        __syncthreads();
        #pragma unroll
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          int src_1 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_8 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid = (((i_8 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
          float condval_7;
          if (valid) {
            condval_7 = ((float*)buf_dyn_shmem)[((int64_t)src_1)];
          } else {
            condval_7 = -CUDART_INF_F;
          }
          maximum[i_8] = condval_7;
          float condval_8;
          if (valid) {
            condval_8 = ((float*)buf_dyn_shmem)[(((int64_t)src_1) + (int64_t)32)];
          } else {
            condval_8 = 0x0p+0f/*0.000000e+00*/;
          }
          denominator[i_8] = condval_8;
        }
        for (int tile = 0; tile < (((end + 63) - begin) >> 6); ++tile) {
          __syncthreads();
          #pragma unroll
          for (int i_9 = 0; i_9 < 8; ++i_9) {
            int condval_9;
            if ((0 <= ((((tile * 64) + (i_9 * 8)) + (((int)threadIdx.x) >> 4)) + begin))) {
              condval_9 = Indices[((((int)blockIdx.x) * 259) + min(((((tile * 64) + (i_9 * 8)) + (((int)threadIdx.x) >> 4)) + begin), 258))];
            } else {
              condval_9 = 0;
            }
            int token = condval_9;
            bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_10;
            if ((((((((tile * 64) + (i_9 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 259))) {
              condval_10 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_10 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_9 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_10;
            bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_11;
            if ((((((((tile * 64) + (i_9 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 259))) {
              condval_11 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_11 = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_9 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_11;
          }
          #pragma unroll
          for (int i_10 = 0; i_10 < 2; ++i_10) {
            float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores + (i_10 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
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
          for (int i_s = 0; i_s < 8; ++i_s) {
            bool valid_1 = (((((i_s & 3) >> 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
            float condval_12;
            if ((valid_1 && (((((((tile * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_s >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin) + (i_s & 1)) < end))) {
              condval_12 = (scores[i_s] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_12 = -CUDART_INF_F;
            }
            scores[i_s] = condval_12;
          }
          *(float2*)(previous + 0) = *(float2*)(maximum + 0);
          __syncthreads();
          #pragma unroll
          for (int i_11 = 0; i_11 < 2; ++i_11) {
            maximum_clear[i_11] = -CUDART_INF_F;
            #pragma unroll
            for (int rv = 0; rv < 4; ++rv) {
              maximum_clear[i_11] = max(maximum_clear[i_11], scores[((((rv & 1) * 4) + (i_11 * 2)) + (rv >> 1))]);
            }
            maximum_clear[i_11] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_11], (&(((float*)buf_dyn_shmem)[11712])));
            maximum_clear[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_11]);
            maximum[i_11] = max(maximum[i_11], maximum_clear[i_11]);
          }
          #pragma unroll
          for (int i_12 = 0; i_12 < 2; ++i_12) {
            float condval_13;
            if ((previous[i_12] == -CUDART_INF_F)) {
              condval_13 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_13 = exp2f(((previous[i_12] - maximum[i_12]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor[i_12] = condval_13;
          }
          #pragma unroll
          for (int i_13 = 0; i_13 < 8; ++i_13) {
            float condval_14;
            if ((scores[i_13] == -CUDART_INF_F)) {
              condval_14 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_14 = exp2f(((scores[i_13] - maximum[((i_13 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores[i_13] = condval_14;
          }
          __syncthreads();
          #pragma unroll
          for (int i_14 = 0; i_14 < 2; ++i_14) {
            subtotal[i_14] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
              subtotal[i_14] = (subtotal[i_14] + scores[((((rv_1 & 1) * 4) + (i_14 * 2)) + (rv_1 >> 1))]);
            }
            subtotal[i_14] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal[i_14], (&(((float*)buf_dyn_shmem)[11456])));
            subtotal[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal[i_14]);
          }
          #pragma unroll
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            denominator[i_15] = ((denominator[i_15] * factor[i_15]) + subtotal[i_15]);
          }
          #pragma unroll
          for (int i_16 = 0; i_16 < 16; ++i_16) {
            out[i_16] = (out[i_16] * factor[((i_16 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_17 = 0; i_17 < 4; ++i_17) {
            uint1 __1;
            float2 v_ = *(float2*)(scores + (i_17 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
            *(uint1*)(probs_local_cast + 0) = __1;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_17 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_17 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_17 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast + 0);
          }
          {
            bfloat16_t A_local_1[8];
            bfloat16_t B_local_1[16];
            __syncthreads();
            for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_1[0])));
              for (int i_18 = 0; i_18 < 2; ++i_18) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_18) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_1[(i_18 * 8)])));
              }
              for (int j = 0; j < 2; ++j) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_19 = 0; i_19 < 8; ++i_19) {
          OUT0[i_19] = out[(((i_19 >> 1) * 4) + (i_19 & 1))];
        }
        #pragma unroll
        for (int i_20 = 0; i_20 < 8; ++i_20) {
          OUT1[i_20] = out[((((i_20 >> 1) * 4) + (i_20 & 1)) + 2)];
        }
        if (Owner[((((int)blockIdx.x) * 168) + (r * 24))] != 0) {
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            cu0[i_21] = OUT0[i_21];
          }
        }
        if (Owner[(((((int)blockIdx.x) * 168) + (r * 24)) + 8)] != 0) {
          #pragma unroll
          for (int i_22 = 0; i_22 < 8; ++i_22) {
            float condval_15;
            if (((label & 1) == 0)) {
              condval_15 = OUT0[i_22];
            } else {
              condval_15 = OUT1[i_22];
            }
            cu1[i_22] = condval_15;
          }
        }
        if (Owner[(((((int)blockIdx.x) * 168) + (r * 24)) + 16)] != 0) {
          #pragma unroll
          for (int i_23 = 0; i_23 < 8; ++i_23) {
            float condval_16;
            if ((((label & 1) + ((label >> 1) & 1)) == 0)) {
              condval_16 = OUT0[i_23];
            } else {
              condval_16 = OUT1[i_23];
            }
            cu2[i_23] = condval_16;
          }
        }
        __syncthreads();
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_24 = 0; i_24 < 2; ++i_24) {
            int src_2 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_24 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
            if (((i_24 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]) {
              ((float*)buf_dyn_shmem)[((int64_t)src_2)] = maximum[i_24];
              ((float*)buf_dyn_shmem)[(((int64_t)src_2) + (int64_t)32)] = denominator[i_24];
            }
          }
        }
      }
    } else {
      int begin_1 = ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 8) + r)]);
      int end_1 = ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 8) + r)]);
      int label_1 = Labels[r];
      if (begin_1 < end_1) {
        #pragma unroll
        for (int i_25 = 0; i_25 < 4; ++i_25) {
          int src_3 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_25 * 8)) + (((int)threadIdx.x) >> 4))];
          bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_17;
          if (((((i_25 * 8) + (((int)threadIdx.x) >> 4)) < Count[((((int)blockIdx.x) * 7) + r)]) && (0 <= src_3))) {
            condval_17 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (min(((int64_t)src_3), (int64_t)23) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_17 = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_25 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_17;
        }
        #pragma unroll
        for (int i_26 = 0; i_26 < 8; ++i_26) {
          float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(out_1 + (i_26 * 4)) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
        }
        if (8 <= Count[((((int)blockIdx.x) * 7) + r)]) {
          int request_2 = (RowMap[((((int)blockIdx.x) * 224) + (r * 32))] >> 3);
          #pragma unroll
          for (int i_27 = 0; i_27 < 32; ++i_27) {
            if ((((i_27 >> 4) * 2) + ((i_27 & 3) >> 1)) < 1) {
              float condval_18;
              if ((request_2 == 0)) {
                condval_18 = cu0[((((i_27 & 15) >> 2) * 2) + (i_27 & 1))];
              } else {
                float condval_19;
                if ((request_2 == 1)) {
                  condval_19 = cu1[((((i_27 & 15) >> 2) * 2) + (i_27 & 1))];
                } else {
                  float condval_20;
                  if ((request_2 == 2)) {
                    condval_20 = cu2[((((i_27 & 15) >> 2) * 2) + (i_27 & 1))];
                  } else {
                    condval_20 = cu3[((((i_27 & 15) >> 2) * 2) + (i_27 & 1))];
                  }
                  condval_19 = condval_20;
                }
                condval_18 = condval_19;
              }
              out_1[i_27] = condval_18;
            }
          }
        }
        if (16 <= Count[((((int)blockIdx.x) * 7) + r)]) {
          int request_3 = (RowMap[(((((int)blockIdx.x) * 224) + (r * 32)) + 8)] >> 3);
          #pragma unroll
          for (int i_28 = 0; i_28 < 8; ++i_28) {
            float condval_21;
            if ((request_3 == 0)) {
              condval_21 = cu0[i_28];
            } else {
              float condval_22;
              if ((request_3 == 1)) {
                condval_22 = cu1[i_28];
              } else {
                float condval_23;
                if ((request_3 == 2)) {
                  condval_23 = cu2[i_28];
                } else {
                  condval_23 = cu3[i_28];
                }
                condval_22 = condval_23;
              }
              condval_21 = condval_22;
            }
            out_1[((((i_28 >> 1) * 4) + (i_28 & 1)) + 2)] = condval_21;
          }
        }
        if (24 <= Count[((((int)blockIdx.x) * 7) + r)]) {
          int request_4 = (RowMap[(((((int)blockIdx.x) * 224) + (r * 32)) + 16)] >> 3);
          #pragma unroll
          for (int i_29 = 0; i_29 < 8; ++i_29) {
            float condval_24;
            if ((request_4 == 0)) {
              condval_24 = cu0[i_29];
            } else {
              float condval_25;
              if ((request_4 == 1)) {
                condval_25 = cu1[i_29];
              } else {
                float condval_26;
                if ((request_4 == 2)) {
                  condval_26 = cu2[i_29];
                } else {
                  condval_26 = cu3[i_29];
                }
                condval_25 = condval_26;
              }
              condval_24 = condval_25;
            }
            out_1[((((i_29 >> 1) * 4) + (i_29 & 1)) + 16)] = condval_24;
          }
        }
        if (32 <= Count[((((int)blockIdx.x) * 7) + r)]) {
          int request_5 = (RowMap[(((((int)blockIdx.x) * 224) + (r * 32)) + 24)] >> 3);
          #pragma unroll
          for (int i_30 = 0; i_30 < 8; ++i_30) {
            float condval_27;
            if ((request_5 == 0)) {
              condval_27 = cu0[i_30];
            } else {
              float condval_28;
              if ((request_5 == 1)) {
                condval_28 = cu1[i_30];
              } else {
                float condval_29;
                if ((request_5 == 2)) {
                  condval_29 = cu2[i_30];
                } else {
                  condval_29 = cu3[i_30];
                }
                condval_28 = condval_29;
              }
              condval_27 = condval_28;
            }
            out_1[(((((((((int)threadIdx.x) & 31) + 96) >> 6) * 16) + ((i_30 >> 1) * 4)) + (i_30 & 1)) + 2)] = condval_27;
          }
        }
        __syncthreads();
        #pragma unroll
        for (int i_31 = 0; i_31 < 4; ++i_31) {
          int src_4 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_31 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_2 = (((i_31 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
          float condval_30;
          if (valid_2) {
            condval_30 = ((float*)buf_dyn_shmem)[((int64_t)src_4)];
          } else {
            condval_30 = -CUDART_INF_F;
          }
          maximum_1[i_31] = condval_30;
          float condval_31;
          if (valid_2) {
            condval_31 = ((float*)buf_dyn_shmem)[(((int64_t)src_4) + (int64_t)32)];
          } else {
            condval_31 = 0x0p+0f/*0.000000e+00*/;
          }
          denominator_1[i_31] = condval_31;
        }
        for (int tile_1 = 0; tile_1 < (((end_1 + 63) - begin_1) >> 6); ++tile_1) {
          __syncthreads();
          #pragma unroll
          for (int i_32 = 0; i_32 < 8; ++i_32) {
            int condval_32;
            if ((0 <= ((((tile_1 * 64) + (i_32 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1))) {
              condval_32 = Indices[((((int)blockIdx.x) * 259) + min(((((tile_1 * 64) + (i_32 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1), 258))];
            } else {
              condval_32 = 0;
            }
            int token_1 = condval_32;
            bfloat16_t broadcast_var_11 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_33;
            if ((((((((tile_1 * 64) + (i_32 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 259))) {
              condval_33 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_33 = make_uint4(__pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_32 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_33;
            bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_34;
            if ((((((((tile_1 * 64) + (i_32 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 259))) {
              condval_34 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_34 = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_32 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_34;
          }
          #pragma unroll
          for (int i_33 = 0; i_33 < 4; ++i_33) {
            float broadcast_var_13 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_1 + (i_33 * 4)) = make_float4(broadcast_var_13, broadcast_var_13, broadcast_var_13, broadcast_var_13);
          }
          {
            bfloat16_t A_local_2[16];
            bfloat16_t B_local_2[8];
            __syncthreads();
            for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
              for (int i_34 = 0; i_34 < 2; ++i_34) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_2 >> 2) * 2048) + (i_34 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(A_local_2[(i_34 * 8)])));
              }
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)])), (&(B_local_2[0])));
              for (int i_35 = 0; i_35 < 2; ++i_35) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + (i_35 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + (i_35 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + ((i_35 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_35 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 4));
              }
            }
          }
          #pragma unroll
          for (int i_36 = 0; i_36 < 2; ++i_36) {
            for (int vec_s = 0; vec_s < 8; ++vec_s) {
              bool valid_3 = ((((i_36 * 16) + (((vec_s & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
              float condval_35;
              if ((valid_3 && (((((((tile_1 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((vec_s >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_1) + (vec_s & 1)) < end_1))) {
                condval_35 = (scores_1[((i_36 * 8) + vec_s)] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
              } else {
                condval_35 = -CUDART_INF_F;
              }
              scores_1[((i_36 * 8) + vec_s)] = condval_35;
            }
          }
          *(float4*)(previous_1 + 0) = *(float4*)(maximum_1 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_37 = 0; i_37 < 4; ++i_37) {
            maximum_clear_1[i_37] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
              maximum_clear_1[i_37] = max(maximum_clear_1[i_37], scores_1[(((((i_37 >> 1) * 8) + ((rv_2 & 1) * 4)) + ((i_37 & 1) * 2)) + (rv_2 >> 1))]);
            }
            maximum_clear_1[i_37] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_37], (&(((float*)buf_dyn_shmem)[11584])));
            maximum_clear_1[i_37] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_37]);
            maximum_1[i_37] = max(maximum_1[i_37], maximum_clear_1[i_37]);
          }
          #pragma unroll
          for (int i_38 = 0; i_38 < 4; ++i_38) {
            float condval_36;
            if ((previous_1[i_38] == -CUDART_INF_F)) {
              condval_36 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_36 = exp2f(((previous_1[i_38] - maximum_1[i_38]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_1[i_38] = condval_36;
          }
          #pragma unroll
          for (int i_39 = 0; i_39 < 16; ++i_39) {
            float condval_37;
            if ((scores_1[i_39] == -CUDART_INF_F)) {
              condval_37 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_37 = exp2f(((scores_1[i_39] - maximum_1[(((i_39 >> 3) * 2) + ((i_39 & 3) >> 1))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_1[i_39] = condval_37;
          }
          __syncthreads();
          #pragma unroll
          for (int i_40 = 0; i_40 < 4; ++i_40) {
            subtotal_1[i_40] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
              subtotal_1[i_40] = (subtotal_1[i_40] + scores_1[(((((i_40 >> 1) * 8) + ((rv_3 & 1) * 4)) + ((i_40 & 1) * 2)) + (rv_3 >> 1))]);
            }
            subtotal_1[i_40] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_40], (&(((float*)buf_dyn_shmem)[11328])));
            subtotal_1[i_40] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_40]);
          }
          #pragma unroll
          for (int i_41 = 0; i_41 < 4; ++i_41) {
            denominator_1[i_41] = ((denominator_1[i_41] * factor_1[i_41]) + subtotal_1[i_41]);
          }
          #pragma unroll
          for (int i_42 = 0; i_42 < 32; ++i_42) {
            out_1[i_42] = (out_1[i_42] * factor_1[(((i_42 >> 4) * 2) + ((i_42 & 3) >> 1))]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_43 = 0; i_43 < 2; ++i_43) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_43 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), __pack_half2(((bfloat16_t)scores_1[(i_43 * 8)]), ((bfloat16_t)scores_1[((i_43 * 8) + 1)])), __pack_half2(((bfloat16_t)scores_1[((i_43 * 8) + 2)]), ((bfloat16_t)scores_1[((i_43 * 8) + 3)])), __pack_half2(((bfloat16_t)scores_1[((i_43 * 8) + 4)]), ((bfloat16_t)scores_1[((i_43 * 8) + 5)])), __pack_half2(((bfloat16_t)scores_1[((i_43 * 8) + 6)]), ((bfloat16_t)scores_1[((i_43 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_3[16];
            bfloat16_t B_local_3[16];
            __syncthreads();
            for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
              for (int i_44 = 0; i_44 < 2; ++i_44) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_44 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(A_local_3[(i_44 * 8)])));
              }
              for (int i_45 = 0; i_45 < 2; ++i_45) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_45) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8320)])), (&(B_local_3[(i_45 * 8)])));
              }
              for (int i_46 = 0; i_46 < 2; ++i_46) {
                for (int j_1 = 0; j_1 < 2; ++j_1) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + ((i_46 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_46 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_1 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + (((i_46 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_46 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_1 * 8) + 4)));
                }
              }
            }
          }
        }
        #pragma unroll
        for (int i_47 = 0; i_47 < 8; ++i_47) {
          OUT0_1[i_47] = out_1[(((i_47 >> 1) * 4) + (i_47 & 1))];
        }
        #pragma unroll
        for (int i_48 = 0; i_48 < 8; ++i_48) {
          OUT1_1[i_48] = out_1[((((i_48 >> 1) * 4) + (i_48 & 1)) + 2)];
        }
        #pragma unroll
        for (int i_49 = 0; i_49 < 8; ++i_49) {
          OUT2[i_49] = out_1[((((i_49 >> 1) * 4) + (i_49 & 1)) + 16)];
        }
        #pragma unroll
        for (int i_50 = 0; i_50 < 8; ++i_50) {
          OUT3[i_50] = out_1[(((((((((int)threadIdx.x) & 31) + 96) >> 6) * 16) + ((i_50 >> 1) * 4)) + (i_50 & 1)) + 2)];
        }
        if (Owner[((((int)blockIdx.x) * 168) + (r * 24))] != 0) {
          #pragma unroll
          for (int i_51 = 0; i_51 < 8; ++i_51) {
            cu0[i_51] = OUT0_1[i_51];
          }
        }
        if (Owner[(((((int)blockIdx.x) * 168) + (r * 24)) + 8)] != 0) {
          #pragma unroll
          for (int i_52 = 0; i_52 < 8; ++i_52) {
            float condval_38;
            if (((label_1 & 1) == 0)) {
              condval_38 = OUT0_1[i_52];
            } else {
              float condval_39;
              if (((label_1 & 1) == 1)) {
                condval_39 = OUT1_1[i_52];
              } else {
                condval_39 = OUT3[i_52];
              }
              condval_38 = condval_39;
            }
            cu1[i_52] = condval_38;
          }
        }
        if (Owner[(((((int)blockIdx.x) * 168) + (r * 24)) + 16)] != 0) {
          #pragma unroll
          for (int i_53 = 0; i_53 < 8; ++i_53) {
            float condval_40;
            if ((((label_1 & 1) + ((label_1 >> 1) & 1)) == 0)) {
              condval_40 = OUT0_1[i_53];
            } else {
              float condval_41;
              if ((((label_1 & 1) + ((label_1 >> 1) & 1)) == 1)) {
                condval_41 = OUT1_1[i_53];
              } else {
                float condval_42;
                if ((((label_1 & 1) + ((label_1 >> 1) & 1)) == 2)) {
                  condval_42 = OUT2[i_53];
                } else {
                  condval_42 = OUT3[i_53];
                }
                condval_41 = condval_42;
              }
              condval_40 = condval_41;
            }
            cu2[i_53] = condval_40;
          }
        }
        __syncthreads();
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_54 = 0; i_54 < 4; ++i_54) {
            int src_5 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_54 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
            if (((i_54 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]) {
              ((float*)buf_dyn_shmem)[((int64_t)src_5)] = maximum_1[i_54];
              ((float*)buf_dyn_shmem)[(((int64_t)src_5) + (int64_t)32)] = denominator_1[i_54];
            }
          }
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i_55 = 0; i_55 < 4; ++i_55) {
    *(float2*)(U + ((((((((int)blockIdx.x) * 196608) + (((int)blockIdx.y) * 3072)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (i_55 * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(cu0 + (i_55 * 2));
  }
  #pragma unroll
  for (int i_56 = 0; i_56 < 4; ++i_56) {
    *(float2*)(U + (((((((((int)blockIdx.x) * 196608) + (((int)blockIdx.y) * 3072)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (i_56 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(float2*)(cu1 + (i_56 * 2));
  }
  #pragma unroll
  for (int i_57 = 0; i_57 < 4; ++i_57) {
    *(float2*)(U + (((((((((int)blockIdx.x) * 196608) + (((int)blockIdx.y) * 3072)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (i_57 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(float2*)(cu2 + (i_57 * 2));
  }
  if (((int)threadIdx.x) < 24) {
    Mout[(((((int)blockIdx.x) * 1536) + (((int)blockIdx.y) * 24)) + ((int)threadIdx.x))] = ((float*)buf_dyn_shmem)[((int)threadIdx.x)];
    Lout[(((((int)blockIdx.x) * 1536) + (((int)blockIdx.y) * 24)) + ((int)threadIdx.x))] = ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)];
  }
}

