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

extern "C" __global__ void main_kernel(const int* __restrict__ Count, const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const bfloat16_t* __restrict__ Q, const int* __restrict__ RowMap, float* __restrict__ U, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ Count, const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const bfloat16_t* __restrict__ Q, const int* __restrict__ RowMap, float* __restrict__ U, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float cu_local_cast[4];
  float out[16];
  float maximum[2];
  float denominator[2];
  float out_1[32];
  float maximum_1[2];
  float denominator_1[2];
  float scores[4];
  float previous[2];
  float maximum_clear[2];
  float factor[2];
  float subtotal[2];
  bfloat16_t probs_local_cast_1[2];
  float scores_1[8];
  float previous_1[2];
  float maximum_clear_1[2];
  float factor_1[2];
  float subtotal_1[2];
  if (((int)threadIdx.x) < 32) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)] = -CUDART_INF_F;
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu_local_cast + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    *(float4*)(((float*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 64)) = *(float4*)(cu_local_cast + 0);
  }
  if (Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))] <= 16) {
    int begin = ((((Offsets[(((((int)blockIdx.x) * 7) + ((int)blockIdx.z)) + 1)] - Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]) * ((int)blockIdx.y)) >> 5) + Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]);
    int end = ((((Offsets[(((((int)blockIdx.x) * 7) + ((int)blockIdx.z)) + 1)] - Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]) * (((int)blockIdx.y) + 1)) >> 5) + Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]);
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      int src = RowMap[((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (i_1 * 8)) + (((int)threadIdx.x) >> 4))];
      bool valid = (((i_1 * 8) + (((int)threadIdx.x) >> 4)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
      bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval;
      if (((valid && (0 <= src)) && (src < 32))) {
        condval = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int64_t)src) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12416)) = condval;
    }
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      int src_1 = RowMap[((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      bool valid_1 = ((((i_2 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      float2 condval_1;
      if (valid_1) {
        condval_1 = *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)64));
      } else {
        condval_1 = make_float2(broadcast_var_2, broadcast_var_2);
      }
      *(float2*)(out + (i_2 * 2)) = condval_1;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      int src_2 = RowMap[((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      bool valid_2 = (((i_3 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
      float condval_2;
      if (valid_2) {
        condval_2 = ((float*)buf_dyn_shmem)[(((int64_t)src_2) + (int64_t)32)];
      } else {
        condval_2 = -CUDART_INF_F;
      }
      maximum[i_3] = condval_2;
      float condval_3;
      if (valid_2) {
        condval_3 = ((float*)buf_dyn_shmem)[((int64_t)src_2)];
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      denominator[i_3] = condval_3;
    }
    for (int tile = 0; tile < (((end + 31) - begin) >> 5); ++tile) {
      __syncthreads();
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        int condval_4;
        if ((0 <= ((((tile * 32) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) + begin))) {
          condval_4 = Indices[((((int)blockIdx.x) * 65536) + min(((((tile * 32) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) + begin), 65535))];
        } else {
          condval_4 = 0;
        }
        int token = condval_4;
        bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_5;
        if ((((((((tile * 32) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 65536))) {
          condval_5 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)8388608) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_5 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_5;
        bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_6;
        if ((((((((tile * 32) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 65536))) {
          condval_6 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)8388608) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_6;
      }
      float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
      {
        bfloat16_t A_local[8];
        bfloat16_t B_local[4];
        __syncthreads();
        for (int ki = 0; ki < 8; ++ki) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12416)])), (&(A_local[0])));
          tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)])), (&(B_local[0])));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        }
      }
      for (int i_s = 0; i_s < 4; ++i_s) {
        bool valid_3 = ((((i_s >> 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
        float condval_7;
        if ((valid_3 && ((((((tile * 32) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin) + (i_s & 1)) < end))) {
          condval_7 = (scores[i_s] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
        } else {
          condval_7 = -CUDART_INF_F;
        }
        scores[i_s] = condval_7;
      }
      *(float2*)(previous + 0) = *(float2*)(maximum + 0);
      __syncthreads();
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        maximum_clear[i_5] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          maximum_clear[i_5] = max(maximum_clear[i_5], scores[((i_5 * 2) + rv)]);
        }
        maximum_clear[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_5], (&(((float*)buf_dyn_shmem)[10944])));
        maximum_clear[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_5]);
        maximum[i_5] = max(maximum[i_5], maximum_clear[i_5]);
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        float condval_8;
        if ((previous[i_6] == -CUDART_INF_F)) {
          condval_8 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_8 = exp2f(((previous[i_6] - maximum[i_6]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
        }
        factor[i_6] = condval_8;
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        float condval_9;
        if ((scores[i_7] == -CUDART_INF_F)) {
          condval_9 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_9 = exp2f(((scores[i_7] - maximum[(i_7 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
        }
        scores[i_7] = condval_9;
      }
      __syncthreads();
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        subtotal[i_8] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
          subtotal[i_8] = (subtotal[i_8] + scores[((i_8 * 2) + rv_1)]);
        }
        subtotal[i_8] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal[i_8], (&(((float*)buf_dyn_shmem)[11200])));
        subtotal[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        denominator[i_9] = ((denominator[i_9] * factor[i_9]) + subtotal[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 16; ++i_10) {
        out[i_10] = (out[i_10] * factor[((i_10 & 3) >> 1)]);
      }
      __syncthreads();
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        uint1 __1;
        float2 v_ = *(float2*)(scores + (i_11 * 2));
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
        *(uint1*)(probs_local_cast_1 + 0) = __1;
        *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_11 * 256) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 20608)) = *(uint1*)(probs_local_cast_1 + 0);
      }
      {
        bfloat16_t A_local_1[8];
        bfloat16_t B_local_1[16];
        __syncthreads();
        for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 20608)])), (&(A_local_1[0])));
          for (int i_12 = 0; i_12 < 2; ++i_12) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_12) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(B_local_1[(i_12 * 8)])));
          }
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      int src_3 = RowMap[((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      bool valid_4 = ((((i_13 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
      if (valid_4) {
        *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_3) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)i_13) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)64)) = *(float2*)(out + (i_13 * 2));
      }
    }
    __syncthreads();
    if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        int src_4 = RowMap[((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (i_14 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
        bool valid_5 = (((i_14 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
        if (valid_5) {
          ((float*)buf_dyn_shmem)[(((int64_t)src_4) + (int64_t)32)] = maximum[i_14];
          ((float*)buf_dyn_shmem)[((int64_t)src_4)] = denominator[i_14];
        }
      }
    }
  } else {
    int begin_1 = ((((Offsets[(((((int)blockIdx.x) * 7) + ((int)blockIdx.z)) + 1)] - Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]) * ((int)blockIdx.y)) >> 5) + Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]);
    int end_1 = ((((Offsets[(((((int)blockIdx.x) * 7) + ((int)blockIdx.z)) + 1)] - Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]) * (((int)blockIdx.y) + 1)) >> 5) + Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]);
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      int src_5 = RowMap[((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (i_15 * 8)) + (((int)threadIdx.x) >> 4))];
      bool valid_6 = (((i_15 * 8) + (((int)threadIdx.x) >> 4)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
      bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_10;
      if (((valid_6 && (0 <= src_5)) && (src_5 < 32))) {
        condval_10 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int64_t)src_5) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_10 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_15 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12416)) = condval_10;
    }
    __syncthreads();
    #pragma unroll
    for (int i_16 = 0; i_16 < 16; ++i_16) {
      int src_6 = RowMap[(((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_16 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      bool valid_7 = ((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_16 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
      float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
      float2 condval_11;
      if (valid_7) {
        condval_11 = *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_6) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + ((((int64_t)i_16) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)64));
      } else {
        condval_11 = make_float2(broadcast_var_7, broadcast_var_7);
      }
      *(float2*)(out_1 + (i_16 * 2)) = condval_11;
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      int src_7 = RowMap[(((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + (i_17 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      bool valid_8 = ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (i_17 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
      float condval_12;
      if (valid_8) {
        condval_12 = ((float*)buf_dyn_shmem)[(((int64_t)src_7) + (int64_t)32)];
      } else {
        condval_12 = -CUDART_INF_F;
      }
      maximum_1[i_17] = condval_12;
      float condval_13;
      if (valid_8) {
        condval_13 = ((float*)buf_dyn_shmem)[((int64_t)src_7)];
      } else {
        condval_13 = 0x0p+0f/*0.000000e+00*/;
      }
      denominator_1[i_17] = condval_13;
    }
    for (int tile_1 = 0; tile_1 < (((end_1 + 31) - begin_1) >> 5); ++tile_1) {
      __syncthreads();
      #pragma unroll
      for (int i_18 = 0; i_18 < 4; ++i_18) {
        int condval_14;
        if ((0 <= ((((tile_1 * 32) + (i_18 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1))) {
          condval_14 = Indices[((((int)blockIdx.x) * 65536) + min(((((tile_1 * 32) + (i_18 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1), 65535))];
        } else {
          condval_14 = 0;
        }
        int token_1 = condval_14;
        bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_15;
        if ((((((((tile_1 * 32) + (i_18 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 65536))) {
          condval_15 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)8388608) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_15 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_18 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)) = condval_15;
        bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_16;
        if ((((((((tile_1 * 32) + (i_18 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 65536))) {
          condval_16 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)8388608) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_16 = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_18 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16512)) = condval_16;
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(scores_1 + (i_19 * 4)) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
      }
      {
        bfloat16_t A_local_2[8];
        bfloat16_t B_local_2[8];
        __syncthreads();
        for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_2 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12416)])), (&(A_local_2[0])));
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki_2 >> 2) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8320)])), (&(B_local_2[0])));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + 4), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 4));
        }
      }
      for (int i_s_1 = 0; i_s_1 < 8; ++i_s_1) {
        bool valid_9 = ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_s_1 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
        float condval_17;
        if ((valid_9 && (((((((tile_1 * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_s_1 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_1) + (i_s_1 & 1)) < end_1))) {
          condval_17 = (scores_1[i_s_1] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
        } else {
          condval_17 = -CUDART_INF_F;
        }
        scores_1[i_s_1] = condval_17;
      }
      *(float2*)(previous_1 + 0) = *(float2*)(maximum_1 + 0);
      __syncthreads();
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        maximum_clear_1[i_20] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
          maximum_clear_1[i_20] = max(maximum_clear_1[i_20], scores_1[((((rv_2 & 1) * 4) + (i_20 * 2)) + (rv_2 >> 1))]);
        }
        maximum_clear_1[i_20] = tl::AllReduce<tl::MaxOp, 128, 64, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_20], (&(((float*)buf_dyn_shmem)[11072])));
        maximum_clear_1[i_20] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_20]);
        maximum_1[i_20] = max(maximum_1[i_20], maximum_clear_1[i_20]);
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        float condval_18;
        if ((previous_1[i_21] == -CUDART_INF_F)) {
          condval_18 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_18 = exp2f(((previous_1[i_21] - maximum_1[i_21]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
        }
        factor_1[i_21] = condval_18;
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        float condval_19;
        if ((scores_1[i_22] == -CUDART_INF_F)) {
          condval_19 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_19 = exp2f(((scores_1[i_22] - maximum_1[((i_22 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
        }
        scores_1[i_22] = condval_19;
      }
      __syncthreads();
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        subtotal_1[i_23] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
          subtotal_1[i_23] = (subtotal_1[i_23] + scores_1[((((rv_3 & 1) * 4) + (i_23 * 2)) + (rv_3 >> 1))]);
        }
        subtotal_1[i_23] = tl::AllReduce<tl::SumOp, 128, 64, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_23], (&(((float*)buf_dyn_shmem)[10816])));
        subtotal_1[i_23] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_23]);
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        denominator_1[i_24] = ((denominator_1[i_24] * factor_1[i_24]) + subtotal_1[i_24]);
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 32; ++i_25) {
        out_1[i_25] = (out_1[i_25] * factor_1[((i_25 & 3) >> 1)]);
      }
      tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 20608)])), __pack_half2(((bfloat16_t)scores_1[0]), ((bfloat16_t)scores_1[1])), __pack_half2(((bfloat16_t)scores_1[2]), ((bfloat16_t)scores_1[3])), __pack_half2(((bfloat16_t)scores_1[4]), ((bfloat16_t)scores_1[5])), __pack_half2(((bfloat16_t)scores_1[6]), ((bfloat16_t)scores_1[7])));
      {
        bfloat16_t A_local_3[8];
        bfloat16_t B_local_3[32];
        __syncthreads();
        for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 20608)])), (&(A_local_3[0])));
          for (int i_26 = 0; i_26 < 4; ++i_26) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_26 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_26 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16512)])), (&(B_local_3[(i_26 * 8)])));
          }
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_27 = 0; i_27 < 16; ++i_27) {
      int src_8 = RowMap[(((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      bool valid_10 = ((((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
      if (valid_10) {
        *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_8) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + ((((int64_t)i_27) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)64)) = *(float2*)(out_1 + (i_27 * 2));
      }
    }
    __syncthreads();
    if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 0) {
      #pragma unroll
      for (int i_28 = 0; i_28 < 2; ++i_28) {
        int src_9 = RowMap[(((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + (i_28 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
        bool valid_11 = ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (i_28 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 6) + ((int)blockIdx.z))]);
        if (valid_11) {
          ((float*)buf_dyn_shmem)[(((int64_t)src_9) + (int64_t)32)] = maximum_1[i_28];
          ((float*)buf_dyn_shmem)[((int64_t)src_9)] = denominator_1[i_28];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 8; ++i_29) {
    *(float4*)(U + (((((((int)blockIdx.x) * 786432) + (((int)blockIdx.z) * 131072)) + (((int)blockIdx.y) * 4096)) + (i_29 * 512)) + (((int)threadIdx.x) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + (((i_29 * 512) + (((int)threadIdx.x) * 4)) + 64));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(Mout[(((((int)blockIdx.x) * 6144) + (((int)blockIdx.z) * 1024)) + (((int)blockIdx.y) * 32))])), (&(((float*)buf_dyn_shmem)[32])), 128);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
    tl::tma_store((&(Lout[(((((int)blockIdx.x) * 6144) + (((int)blockIdx.z) * 1024)) + (((int)blockIdx.y) * 32))])), (&(((float*)buf_dyn_shmem)[0])), 128);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
}

