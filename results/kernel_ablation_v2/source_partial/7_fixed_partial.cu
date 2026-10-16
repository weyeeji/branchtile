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

extern "C" __global__ void main_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const int* __restrict__ Owner, const bfloat16_t* __restrict__ Q, float* __restrict__ U, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const int* __restrict__ Owner, const bfloat16_t* __restrict__ Q, float* __restrict__ U, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float out[32];
  float maximum[2];
  float denominator[2];
  float scores[16];
  float previous[2];
  float maximum_clear[2];
  float factor[2];
  float subtotal[2];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((int)blockIdx.x) * 4096) + (i * 1024)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(out + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = -CUDART_INF_F;
  *(float2*)(maximum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(denominator + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  int begin = ((((Offsets[(((((int)blockIdx.x) * 7) + ((int)blockIdx.z)) + 1)] - Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]) * ((int)blockIdx.y)) >> 2) + Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]);
  int end = ((((Offsets[(((((int)blockIdx.x) * 7) + ((int)blockIdx.z)) + 1)] - Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]) * (((int)blockIdx.y) + 1)) >> 2) + Offsets[((((int)blockIdx.x) * 7) + ((int)blockIdx.z))]);
  for (int tile = 0; tile < (((end + 63) - begin) >> 6); ++tile) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      int condval;
      if ((0 <= ((((tile * 64) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + begin))) {
        condval = Indices[((((int)blockIdx.x) * 65536) + min(((((tile * 64) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + begin), 65535))];
      } else {
        condval = 0;
      }
      int token = condval;
      bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_1;
      if ((((((((tile * 64) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 65536))) {
        condval_1 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)8388608) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)) = condval_1;
      bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if ((((((((tile * 64) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 65536))) {
        condval_2 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)8388608) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)) = condval_2;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_3 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
    }
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[16];
      __syncthreads();
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local[(i_4 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      for (int vec_s = 0; vec_s < 2; ++vec_s) {
        float condval_3;
        if (((Owner[(((((((int)blockIdx.x) * 192) + (((int)blockIdx.z) * 32)) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] != 0) && (((((((tile * 64) + ((((int)threadIdx.x) >> 6) * 32)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin) + vec_s) < end))) {
          condval_3 = (scores[((i_5 * 2) + vec_s)] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
        } else {
          condval_3 = -CUDART_INF_F;
        }
        scores[((i_5 * 2) + vec_s)] = condval_3;
      }
    }
    *(float2*)(previous + 0) = *(float2*)(maximum + 0);
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      maximum_clear[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        maximum_clear[i_6] = max(maximum_clear[i_6], scores[((((rv & 3) * 4) + (i_6 * 2)) + (rv >> 2))]);
      }
      maximum_clear[i_6] = tl::AllReduce<tl::MaxOp, 128, 64, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_6], (&(((float*)buf_dyn_shmem)[11392])));
      maximum_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_6]);
      maximum[i_6] = max(maximum[i_6], maximum_clear[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      float condval_4;
      if ((previous[i_7] == -CUDART_INF_F)) {
        condval_4 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_4 = exp2f(((previous[i_7] - maximum[i_7]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
      }
      factor[i_7] = condval_4;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      float condval_5;
      if ((scores[i_8] == -CUDART_INF_F)) {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_5 = exp2f(((scores[i_8] - maximum[((i_8 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
      }
      scores[i_8] = condval_5;
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      subtotal[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        subtotal[i_9] = (subtotal[i_9] + scores[((((rv_1 & 3) * 4) + (i_9 * 2)) + (rv_1 >> 2))]);
      }
      subtotal[i_9] = tl::AllReduce<tl::SumOp, 128, 64, 0, tl::NamedBarrier<128>>::run(subtotal[i_9], (&(((float*)buf_dyn_shmem)[11264])));
      subtotal[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      denominator[i_10] = ((denominator[i_10] * factor[i_10]) + subtotal[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      out[i_11] = (out[i_11] * factor[((i_11 & 3) >> 1)]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_12) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20480)])), __pack_half2(((bfloat16_t)scores[(i_12 * 8)]), ((bfloat16_t)scores[((i_12 * 8) + 1)])), __pack_half2(((bfloat16_t)scores[((i_12 * 8) + 2)]), ((bfloat16_t)scores[((i_12 * 8) + 3)])), __pack_half2(((bfloat16_t)scores[((i_12 * 8) + 4)]), ((bfloat16_t)scores[((i_12 * 8) + 5)])), __pack_half2(((bfloat16_t)scores[((i_12 * 8) + 6)]), ((bfloat16_t)scores[((i_12 * 8) + 7)])));
    }
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_1[32];
      __syncthreads();
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20480)])), (&(A_local_1[0])));
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])), (&(B_local_1[(i_13 * 8)])));
        }
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 16; ++i_14) {
    *(float2*)(U + (((((((((((int)blockIdx.x) * 98304) + (((int)blockIdx.z) * 16384)) + (((int)blockIdx.y) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((i_14 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + ((i_14 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(out + (i_14 * 2));
  }
  if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 0) {
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      Mout[((((((((int)blockIdx.x) * 768) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + (i_15 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = maximum[i_15];
      Lout[((((((((int)blockIdx.x) * 768) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + (i_15 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = denominator[i_15];
    }
  }
}

